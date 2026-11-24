// Package weights 读取 PyTorch 的 .pth/.pth.tar（zip 序列化）权重文件。
//
// 支持的子集：pickle protocol 2..4 的常用 opcode、torch._utils._rebuild_tensor_v2
// 与 _rebuild_parameter、collections.OrderedDict、persistent_id 形式的 storage 引用。
// 只物化被请求的浮点张量；非浮点（如 int64 的 num_batches_tracked）会被跳过。
//
// 本包不做布局转换，返回的张量保持权重文件里的原始 layout（如 conv 的 OIHW）。
package weights

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
)

// Dtype 是权重文件中的元素类型。
type Dtype int

const (
	Float32 Dtype = iota
	Float64
	Float16
	Int64
	Other
)

func (d Dtype) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Float16:
		return "float16"
	case Int64:
		return "int64"
	}
	return "other"
}

func (d Dtype) size() int {
	switch d {
	case Float32:
		return 4
	case Float64:
		return 8
	case Float16:
		return 2
	case Int64:
		return 8
	}
	return 0
}

// Tensor 是权重文件中的一个张量（已按 Shape 物化为连续 row-major）。
type Tensor struct {
	Shape []int
	Data  []float32
}

// Len 返回元素个数。
func (t *Tensor) Len() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// File 是一次解析结果。
type File struct {
	// Meta 保存 pickle 顶层的标量/字符串（如 epoch、best_key），其它类型忽略。
	Meta    map[string]any
	Tensors map[string]*Tensor
	// Skipped 记录未物化的张量名（整型/不支持的 dtype），如 num_batches_tracked。
	Skipped []string
	dtypes  map[string]Dtype
}

// spec 是 pickle 中 _rebuild_tensor_v2 的原始参数。
type spec struct {
	storage string
	offset  int64
	shape   []int64
	stride  []int64
	dtype   Dtype
	name    string
}

type storageRef struct {
	key   string
	dtype Dtype
}

// Open 解析 path 指向的 PyTorch zip 序列化文件。只有 state_dict（若存在）会被物化；
// 需要额外物化时可使用 OpenWithFilter。
func Open(path string) (*File, error) {
	return OpenWithFilter(path, nil)
}

// OpenWithFilter 与 Open 相同，但只物化 filter 返回 true 的张量（filter 为 nil 表示全部物化）。
func OpenWithFilter(path string, filter func(name string) bool) (*File, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("打开权重文件 %s: %w", path, err)
	}
	defer zr.Close()

	var pklName string
	files := make(map[string]*zip.File, len(zr.File))
	for _, f := range zr.File {
		files[f.Name] = f
		if f.Name == "archive/data.pkl" || f.Name == "data.pkl" {
			pklName = f.Name
		}
	}
	if pklName == "" {
		return nil, fmt.Errorf("%s 不是 PyTorch zip 序列化格式（找不到 data.pkl）", path)
	}
	// 张量数据存放在 pickle 同级的 data/ 目录下：<prefix>data.pkl → <prefix>data/<key>
	prefix := pklName[:len(pklName)-len("data.pkl")] + "data/"

	blob, err := readZipFile(files[pklName])
	if err != nil {
		return nil, fmt.Errorf("读取 %s: %w", pklName, err)
	}
	p := &parser{data: blob, memo: map[int]any{}}
	if err := p.run(); err != nil {
		return nil, fmt.Errorf("解析 pickle: %w", err)
	}
	if len(p.stack) == 0 {
		return nil, fmt.Errorf("pickle 解析结果为空")
	}

	f := &File{Meta: map[string]any{}, Tensors: map[string]*Tensor{}, dtypes: map[string]Dtype{}}
	root, ok := p.stack[0].(map[any]any)
	if !ok {
		return nil, fmt.Errorf("pickle 顶层不是 dict（实际 %T）", p.stack[0])
	}

	var specs map[any]any
	if sd, ok := root["state_dict"]; ok {
		m, ok := sd.(map[any]any)
		if !ok {
			return nil, fmt.Errorf("state_dict 不是 dict（实际 %T）", sd)
		}
		specs = m
	} else {
		specs = root
	}
	for k, v := range root {
		if k == "state_dict" {
			continue
		}
		switch tv := v.(type) {
		case string, int64, float64, bool:
			if ks, ok := k.(string); ok {
				f.Meta[ks] = tv
			}
		}
	}

	names := make([]string, 0, len(specs))
	byStorage := map[string][]spec{}
	for k, v := range specs {
		name, ok := k.(string)
		if !ok {
			continue
		}
		s, ok := v.(spec)
		if !ok {
			return nil, fmt.Errorf("state_dict[%s] 不是张量（实际 %T）", name, v)
		}
		s.name = name
		names = append(names, name)
		byStorage[s.storage] = append(byStorage[s.storage], s)
	}
	sort.Strings(names)

	for storage, group := range byStorage {
		// 单个 storage 可能被多个张量引用（这里按 view 提取）；同一 storage 只读一次。
		var wanted []spec
		for _, s := range group {
			if filter == nil || filter(s.name) {
				wanted = append(wanted, s)
			}
		}
		if len(wanted) == 0 {
			continue
		}
		dt := wanted[0].dtype
		raw, err := readStorage(files, prefix+storage, dt)
		if err != nil {
			return nil, fmt.Errorf("读取 storage %s: %w", storage, err)
		}
		if dt == Int64 || dt == Other {
			for _, s := range wanted {
				f.Skipped = append(f.Skipped, s.name)
			}
			continue
		}
		for _, s := range wanted {
			data, err := extract(raw, s, dt)
			if err != nil {
				return nil, fmt.Errorf("张量 %s: %w", s.name, err)
			}
			shape := make([]int, len(s.shape))
			for i, d := range s.shape {
				shape[i] = int(d)
			}
			f.Tensors[s.name] = &Tensor{Shape: shape, Data: data}
			f.dtypes[s.name] = dt
		}
	}
	return f, nil
}

// Names 返回已物化的张量名（升序）。
func (f *File) Names() []string {
	names := make([]string, 0, len(f.Tensors))
	for k := range f.Tensors {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// Get 按名字取张量。
func (f *File) Get(name string) (*Tensor, error) {
	t, ok := f.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("权重中不存在张量 %q", name)
	}
	return t, nil
}

func readZipFile(f *zip.File) ([]byte, error) {
	if f == nil {
		return nil, fmt.Errorf("文件不存在")
	}
	rc, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return io.ReadAll(rc)
}

// readStorage 读取裸 storage 并按 dtype 解码为 float32 视图所需的原始字节。
// 对 int64 返回 nil（不支持物化）。
func readStorage(files map[string]*zip.File, name string, dt Dtype) ([]byte, error) {
	if dt == Int64 || dt == Other {
		return nil, nil
	}
	zf, ok := files[name]
	if !ok {
		return nil, fmt.Errorf("zip 中不存在 %s", name)
	}
	return readZipFile(zf)
}

// extract 依据 offset/shape/stride 从 storage 字节中取连续化后的张量。
func extract(raw []byte, s spec, dt Dtype) ([]float32, error) {
	n := 1
	for _, d := range s.shape {
		n *= int(d)
	}
	if dt == Int64 || dt == Other {
		return nil, nil
	}
	elem := dt.size()
	need := (s.offset + int64(n)) * int64(elem)
	if int64(len(raw)) < need {
		return nil, fmt.Errorf("storage 太短: 需要 %d 字节，实际 %d", need, len(raw))
	}
	out := make([]float32, n)
	base := s.offset * int64(elem)
	if isContiguous(s.shape, s.stride) {
		for i := 0; i < n; i++ {
			out[i] = decodeElem(raw, base+int64(i*elem), dt)
		}
		return out, nil
	}
	idx := make([]int, len(s.shape))
	off := base
	for i := 0; i < n; i++ {
		out[i] = decodeElem(raw, off, dt)
		// 递增多维下标（row-major 顺序遍历目标布局）
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			off += s.stride[d] * int64(elem)
			if idx[d] < int(s.shape[d]) {
				break
			}
			off -= s.stride[d] * int64(s.shape[d]) * int64(elem)
			idx[d] = 0
		}
	}
	return out, nil
}

func isContiguous(shape, stride []int64) bool {
	acc := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] == 1 {
			continue
		}
		if stride[i] != acc {
			return false
		}
		acc *= shape[i]
	}
	return true
}

func decodeElem(raw []byte, off int64, dt Dtype) float32 {
	switch dt {
	case Float32:
		return math.Float32frombits(binary.LittleEndian.Uint32(raw[off:]))
	case Float64:
		return float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[off:])))
	case Float16:
		return float16ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
	}
	return 0
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign << 31
		} else {
			// 次正规数：规格化
			e := uint32(127 - 15 + 1)
			for mant&0x400 == 0 {
				mant <<= 1
				e--
			}
			bits = sign<<31 | e<<23 | (mant&0x3ff)<<13
		}
	case exp == 0x1f:
		bits = sign<<31 | 0xff<<23 | mant<<13
	default:
		bits = sign<<31 | (exp+127-15)<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}

// ---- 迷你 pickle 虚拟机 ----

type parser struct {
	data       []byte
	pos        int
	stack      []any
	marks      []int
	memo       map[int]any
	lastDtype  Dtype
	haveDtype  bool
	lastSymbol string
}

func (p *parser) readByte() byte {
	b := p.data[p.pos]
	p.pos++
	return b
}

func (p *parser) readN(n int) []byte {
	b := p.data[p.pos : p.pos+n]
	p.pos += n
	return b
}

func (p *parser) readLine() []byte {
	s := p.pos
	for p.data[p.pos] != '\n' {
		p.pos++
	}
	p.pos++
	return p.data[s : p.pos-1]
}

func (p *parser) push(v any) { p.stack = append(p.stack, v) }

func (p *parser) pop() any {
	v := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]
	return v
}

func (p *parser) popMark() []any {
	m := p.marks[len(p.marks)-1]
	p.marks = p.marks[:len(p.marks)-1]
	items := make([]any, len(p.stack)-m)
	copy(items, p.stack[m:])
	p.stack = p.stack[:m]
	return items
}

func (p *parser) run() (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("位置 %d 处解码失败: %v", p.pos, r)
		}
	}()
	for p.pos < len(p.data) {
		op := p.readByte()
		switch op {
		case 0x80: // PROTO
			p.readByte()
		case 0x95: // FRAME
			p.readN(8)
		case '.': // STOP
			return nil
		case '(': // MARK
			p.marks = append(p.marks, len(p.stack))
		case '0': // POP
			p.pop()
		case 'N': // NONE
			p.push(nil)
		case 0x88: // NEWTRUE
			p.push(true)
		case 0x89: // NEWFALSE
			p.push(false)
		case '}': // EMPTY_DICT
			p.push(map[any]any{})
		case ')': // EMPTY_TUPLE
			p.push([]any{})
		case ']': // EMPTY_LIST
			p.push([]any{})
		case 'q': // BINPUT
			p.memo[int(p.readByte())] = p.stack[len(p.stack)-1]
		case 'r': // LONG_BINPUT
			p.memo[int(binary.LittleEndian.Uint32(p.readN(4)))] = p.stack[len(p.stack)-1]
		case 'h': // BINGET
			p.push(p.memo[int(p.readByte())])
		case 'j': // LONG_BINGET
			p.push(p.memo[int(binary.LittleEndian.Uint32(p.readN(4)))])
		case 0x94: // MEMOIZE
			p.memo[len(p.memo)] = p.stack[len(p.stack)-1]
		case 0x8c: // SHORT_BINUNICODE
			p.push(string(p.readN(int(p.readByte()))))
		case 'X': // BINUNICODE
			p.push(string(p.readN(int(binary.LittleEndian.Uint32(p.readN(4))))))
		case 'T': // BINSTRING
			p.push(string(p.readN(int(binary.LittleEndian.Uint32(p.readN(4))))))
		case 'U': // SHORT_BINSTRING
			p.push(string(p.readN(int(p.readByte()))))
		case 0x8d: // SHORT_BINBYTES
			p.push([]byte(p.readN(int(p.readByte()))))
		case 'B': // BINBYTES
			p.push([]byte(p.readN(int(binary.LittleEndian.Uint32(p.readN(4))))))
		case 'K': // BININT1
			p.push(int64(p.readByte()))
		case 'M': // BININT2
			p.push(int64(binary.LittleEndian.Uint16(p.readN(2))))
		case 'J': // BININT
			p.push(int64(int32(binary.LittleEndian.Uint32(p.readN(4)))))
		case 'G': // BINFLOAT
			p.push(math.Float64frombits(binary.BigEndian.Uint64(p.readN(8))))
		case 'I': // INT
			s := string(p.readLine())
			switch s {
			case "01":
				p.push(true)
			case "00":
				p.push(false)
			default:
				v, perr := strconv.ParseInt(s, 10, 64)
				if perr != nil {
					return fmt.Errorf("INT 解析失败 %q: %w", s, perr)
				}
				p.push(v)
			}
		case 'L': // LONG
			s := string(p.readLine())
			v, perr := strconv.ParseInt(s[:len(s)-1], 10, 64)
			if perr != nil {
				return fmt.Errorf("LONG 解析失败 %q: %w", s, perr)
			}
			p.push(v)
		case 'F': // FLOAT
			v, perr := strconv.ParseFloat(string(p.readLine()), 64)
			if perr != nil {
				return fmt.Errorf("FLOAT 解析失败: %w", perr)
			}
			p.push(v)
		case 'S': // STRING
			s := string(p.readLine())
			p.push(s[1 : len(s)-1])
		case 'V': // UNICODE
			p.push(string(p.readLine()))
		case 't': // TUPLE
			p.push(p.popMark())
		case 0x85: // TUPLE1
			a := p.pop()
			p.push([]any{a})
		case 0x86: // TUPLE2
			b := p.pop()
			a := p.pop()
			p.push([]any{a, b})
		case 0x87: // TUPLE3
			c := p.pop()
			b := p.pop()
			a := p.pop()
			p.push([]any{a, b, c})
		case 'l': // LIST
			p.push(p.popMark())
		case 'd': // DICT
			items := p.popMark()
			m := make(map[any]any, len(items)/2)
			for i := 0; i+1 < len(items); i += 2 {
				m[items[i]] = items[i+1]
			}
			p.push(m)
		case 'u': // SETITEMS
			items := p.popMark()
			m := p.stack[len(p.stack)-1].(map[any]any)
			for i := 0; i+1 < len(items); i += 2 {
				m[items[i]] = items[i+1]
			}
		case 's': // SETITEM
			v := p.pop()
			k := p.pop()
			p.stack[len(p.stack)-1].(map[any]any)[k] = v
		case 'a': // APPEND
			v := p.pop()
			l := p.stack[len(p.stack)-1].([]any)
			p.stack[len(p.stack)-1] = append(l, v)
		case 'e': // APPENDS
			items := p.popMark()
			l := p.stack[len(p.stack)-1].([]any)
			p.stack[len(p.stack)-1] = append(l, items...)
		case 'c': // GLOBAL
			mod := string(p.readLine())
			name := string(p.readLine())
			p.push(p.symbol(mod + "." + name))
		case 0x93: // STACK_GLOBAL
			name, _ := p.pop().(string)
			mod, _ := p.pop().(string)
			p.push(p.symbol(mod + "." + name))
		case 'R': // REDUCE
			args := p.pop()
			fn := p.pop()
			p.push(p.reduce(fn, args))
		case 0x81: // NEWOBJ
			args := p.pop()
			p.push(p.reduce(p.pop(), args))
		case 'b': // BUILD
			p.pop()
		case 'Q': // BINPERSID
			p.push(p.persistent(p.pop()))
		case 'p': // PUT（文本）
			idx, perr := strconv.Atoi(string(p.readLine()))
			if perr != nil {
				return fmt.Errorf("PUT 解析失败: %w", perr)
			}
			p.memo[idx] = p.stack[len(p.stack)-1]
		case 'g': // GET（文本）
			idx, perr := strconv.Atoi(string(p.readLine()))
			if perr != nil {
				return fmt.Errorf("GET 解析失败: %w", perr)
			}
			p.push(p.memo[idx])
		default:
			return fmt.Errorf("不支持的 opcode 0x%02x (%q)", op, op)
		}
	}
	return nil
}

// symbol 把 GLOBAL 记录为字符串标记，同时记下最近的 storage 类型名。
func (p *parser) symbol(full string) string {
	p.lastSymbol = full
	switch full {
	case "torch.FloatStorage":
		p.lastDtype, p.haveDtype = Float32, true
	case "torch.DoubleStorage":
		p.lastDtype, p.haveDtype = Float64, true
	case "torch.HalfStorage":
		p.lastDtype, p.haveDtype = Float16, true
	case "torch.LongStorage":
		p.lastDtype, p.haveDtype = Int64, true
	case "torch.BFloat16Storage":
		p.lastDtype, p.haveDtype = Other, true
	}
	return full
}

func dtypeOfStorageType(sym string) (Dtype, bool) {
	switch sym {
	case "torch.FloatStorage":
		return Float32, true
	case "torch.DoubleStorage":
		return Float64, true
	case "torch.HalfStorage":
		return Float16, true
	case "torch.LongStorage":
		return Int64, true
	}
	return Other, false
}

// persistent 处理 BINPERSID：参数是 ('storage', StorageType, key, location, numel)。
func (p *parser) persistent(pid any) any {
	tup, ok := pid.([]any)
	if !ok {
		return pid
	}
	var key string
	var dt = Float32
	for _, v := range tup {
		switch tv := v.(type) {
		case string:
			if tv != "storage" && key == "" && !isStorageTypeSym(tv) {
				key = tv
			}
			if d, ok := dtypeOfStorageType(tv); ok {
				dt = d
			}
		}
	}
	if key == "" {
		return pid
	}
	return storageRef{key: key, dtype: dt}
}

func isStorageTypeSym(s string) bool {
	_, ok := dtypeOfStorageType(s)
	return ok
}

func (p *parser) reduce(fn, args any) any {
	name, _ := fn.(string)
	switch name {
	case "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor":
		a, ok := args.([]any)
		if !ok || len(a) < 4 {
			panic(fmt.Sprintf("_rebuild_tensor 参数异常: %T", args))
		}
		s := spec{}
		switch sv := a[0].(type) {
		case storageRef:
			s.storage, s.dtype = sv.key, sv.dtype
		case string:
			s.storage, s.dtype = sv, Float32
			if p.haveDtype {
				s.dtype = p.lastDtype
			}
		default:
			panic(fmt.Sprintf("未知 storage 引用类型 %T", a[0]))
		}
		s.offset = toInt64(a[1])
		s.shape = toInt64Slice(a[2])
		s.stride = toInt64Slice(a[3])
		return s
	case "torch._utils._rebuild_parameter":
		a, ok := args.([]any)
		if !ok || len(a) < 1 {
			panic("_rebuild_parameter 参数异常")
		}
		return a[0]
	case "collections.OrderedDict":
		return map[any]any{}
	case "numpy.core.multiarray.scalar", "numpy.dtype":
		return name
	}
	return name
}

func toInt64(v any) int64 {
	switch tv := v.(type) {
	case int64:
		return tv
	case int:
		return int64(tv)
	}
	panic(fmt.Sprintf("期望整数，实际 %T", v))
}

func toInt64Slice(v any) []int64 {
	a, ok := v.([]any)
	if !ok {
		panic(fmt.Sprintf("期望整数序列，实际 %T", v))
	}
	out := make([]int64, len(a))
	for i, e := range a {
		out[i] = toInt64(e)
	}
	return out
}

// Size 返回权重文件路径对应大小，供 CLI 提示用。
func Size(path string) (int64, error) {
	st, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return st.Size(), nil
}
