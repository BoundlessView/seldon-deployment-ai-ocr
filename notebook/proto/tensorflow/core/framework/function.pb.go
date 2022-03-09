// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/framework/function.proto

package framework

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

// A library is a set of named functions.
type FunctionDefLibrary struct {
	Function             []*FunctionDef `protobuf:"bytes,1,rep,name=function,proto3" json:"function,omitempty"`
	Gradient             []*GradientDef `protobuf:"bytes,2,rep,name=gradient,proto3" json:"gradient,omitempty"`
	XXX_NoUnkeyedLiteral struct{}       `json:"-"`
	XXX_unrecognized     []byte         `json:"-"`
	XXX_sizecache        int32          `json:"-"`
}

func (m *FunctionDefLibrary) Reset()         { *m = FunctionDefLibrary{} }
func (m *FunctionDefLibrary) String() string { return proto.CompactTextString(m) }
func (*FunctionDefLibrary) ProtoMessage()    {}
func (*FunctionDefLibrary) Descriptor() ([]byte, []int) {
	return fileDescriptor_507748d6812c5f14, []int{0}
}

func (m *FunctionDefLibrary) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_FunctionDefLibrary.Unmarshal(m, b)
}
func (m *FunctionDefLibrary) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_FunctionDefLibrary.Marshal(b, m, deterministic)
}
func (m *FunctionDefLibrary) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FunctionDefLibrary.Merge(m, src)
}
func (m *FunctionDefLibrary) XXX_Size() int {
	return xxx_messageInfo_FunctionDefLibrary.Size(m)
}
func (m *FunctionDefLibrary) XXX_DiscardUnknown() {
	xxx_messageInfo_FunctionDefLibrary.DiscardUnknown(m)
}

var xxx_messageInfo_FunctionDefLibrary proto.InternalMessageInfo

func (m *FunctionDefLibrary) GetFunction() []*FunctionDef {
	if m != nil {
		return m.Function
	}
	return nil
}

func (m *FunctionDefLibrary) GetGradient() []*GradientDef {
	if m != nil {
		return m.Gradient
	}
	return nil
}

// A function can be instantiated when the runtime can bind every attr
// with a value. When a GraphDef has a call to a function, it must
// have binding for every attr defined in the signature.
//
// TODO(zhifengc):
//   * device spec, etc.
type FunctionDef struct {
	// The definition of the function's name, arguments, return values,
	// attrs etc.
	Signature *OpDef `protobuf:"bytes,1,opt,name=signature,proto3" json:"signature,omitempty"`
	// Attributes specific to this function definition.
	Attr    map[string]*AttrValue            `protobuf:"bytes,5,rep,name=attr,proto3" json:"attr,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	ArgAttr map[uint32]*FunctionDef_ArgAttrs `protobuf:"bytes,7,rep,name=arg_attr,json=argAttr,proto3" json:"arg_attr,omitempty" protobuf_key:"varint,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	// By convention, "op" in node_def is resolved by consulting with a
	// user-defined library first. If not resolved, "func" is assumed to
	// be a builtin op.
	NodeDef []*NodeDef `protobuf:"bytes,3,rep,name=node_def,json=nodeDef,proto3" json:"node_def,omitempty"`
	// A mapping from the output arg names from `signature` to the
	// outputs from `node_def` that should be returned by the function.
	Ret map[string]string `protobuf:"bytes,4,rep,name=ret,proto3" json:"ret,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	// A mapping from control output names from `signature` to node names in
	// `node_def` which should be control outputs of this function.
	ControlRet           map[string]string `protobuf:"bytes,6,rep,name=control_ret,json=controlRet,proto3" json:"control_ret,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *FunctionDef) Reset()         { *m = FunctionDef{} }
func (m *FunctionDef) String() string { return proto.CompactTextString(m) }
func (*FunctionDef) ProtoMessage()    {}
func (*FunctionDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_507748d6812c5f14, []int{1}
}

func (m *FunctionDef) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_FunctionDef.Unmarshal(m, b)
}
func (m *FunctionDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_FunctionDef.Marshal(b, m, deterministic)
}
func (m *FunctionDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FunctionDef.Merge(m, src)
}
func (m *FunctionDef) XXX_Size() int {
	return xxx_messageInfo_FunctionDef.Size(m)
}
func (m *FunctionDef) XXX_DiscardUnknown() {
	xxx_messageInfo_FunctionDef.DiscardUnknown(m)
}

var xxx_messageInfo_FunctionDef proto.InternalMessageInfo

func (m *FunctionDef) GetSignature() *OpDef {
	if m != nil {
		return m.Signature
	}
	return nil
}

func (m *FunctionDef) GetAttr() map[string]*AttrValue {
	if m != nil {
		return m.Attr
	}
	return nil
}

func (m *FunctionDef) GetArgAttr() map[uint32]*FunctionDef_ArgAttrs {
	if m != nil {
		return m.ArgAttr
	}
	return nil
}

func (m *FunctionDef) GetNodeDef() []*NodeDef {
	if m != nil {
		return m.NodeDef
	}
	return nil
}

func (m *FunctionDef) GetRet() map[string]string {
	if m != nil {
		return m.Ret
	}
	return nil
}

func (m *FunctionDef) GetControlRet() map[string]string {
	if m != nil {
		return m.ControlRet
	}
	return nil
}

// Attributes for function arguments. These attributes are the same set of
// valid attributes as to _Arg nodes.
type FunctionDef_ArgAttrs struct {
	Attr                 map[string]*AttrValue `protobuf:"bytes,1,rep,name=attr,proto3" json:"attr,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	XXX_NoUnkeyedLiteral struct{}              `json:"-"`
	XXX_unrecognized     []byte                `json:"-"`
	XXX_sizecache        int32                 `json:"-"`
}

func (m *FunctionDef_ArgAttrs) Reset()         { *m = FunctionDef_ArgAttrs{} }
func (m *FunctionDef_ArgAttrs) String() string { return proto.CompactTextString(m) }
func (*FunctionDef_ArgAttrs) ProtoMessage()    {}
func (*FunctionDef_ArgAttrs) Descriptor() ([]byte, []int) {
	return fileDescriptor_507748d6812c5f14, []int{1, 1}
}

func (m *FunctionDef_ArgAttrs) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_FunctionDef_ArgAttrs.Unmarshal(m, b)
}
func (m *FunctionDef_ArgAttrs) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_FunctionDef_ArgAttrs.Marshal(b, m, deterministic)
}
func (m *FunctionDef_ArgAttrs) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FunctionDef_ArgAttrs.Merge(m, src)
}
func (m *FunctionDef_ArgAttrs) XXX_Size() int {
	return xxx_messageInfo_FunctionDef_ArgAttrs.Size(m)
}
func (m *FunctionDef_ArgAttrs) XXX_DiscardUnknown() {
	xxx_messageInfo_FunctionDef_ArgAttrs.DiscardUnknown(m)
}

var xxx_messageInfo_FunctionDef_ArgAttrs proto.InternalMessageInfo

func (m *FunctionDef_ArgAttrs) GetAttr() map[string]*AttrValue {
	if m != nil {
		return m.Attr
	}
	return nil
}

// GradientDef defines the gradient function of a function defined in
// a function library.
//
// A gradient function g (specified by gradient_func) for a function f
// (specified by function_name) must follow the following:
//
// The function 'f' must be a numerical function which takes N inputs
// and produces M outputs. Its gradient function 'g', which is a
// function taking N + M inputs and produces N outputs.
//
// I.e. if we have
//    (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
// then, g is
//    (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
//                                      dL/dy1, dL/dy2, ..., dL/dy_M),
// where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
// loss function). dL/dx_i is the partial derivative of L with respect
// to x_i.
type GradientDef struct {
	FunctionName         string   `protobuf:"bytes,1,opt,name=function_name,json=functionName,proto3" json:"function_name,omitempty"`
	GradientFunc         string   `protobuf:"bytes,2,opt,name=gradient_func,json=gradientFunc,proto3" json:"gradient_func,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *GradientDef) Reset()         { *m = GradientDef{} }
func (m *GradientDef) String() string { return proto.CompactTextString(m) }
func (*GradientDef) ProtoMessage()    {}
func (*GradientDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_507748d6812c5f14, []int{2}
}

func (m *GradientDef) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_GradientDef.Unmarshal(m, b)
}
func (m *GradientDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_GradientDef.Marshal(b, m, deterministic)
}
func (m *GradientDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_GradientDef.Merge(m, src)
}
func (m *GradientDef) XXX_Size() int {
	return xxx_messageInfo_GradientDef.Size(m)
}
func (m *GradientDef) XXX_DiscardUnknown() {
	xxx_messageInfo_GradientDef.DiscardUnknown(m)
}

var xxx_messageInfo_GradientDef proto.InternalMessageInfo

func (m *GradientDef) GetFunctionName() string {
	if m != nil {
		return m.FunctionName
	}
	return ""
}

func (m *GradientDef) GetGradientFunc() string {
	if m != nil {
		return m.GradientFunc
	}
	return ""
}

func init() {
	proto.RegisterType((*FunctionDefLibrary)(nil), "tensorflow.FunctionDefLibrary")
	proto.RegisterType((*FunctionDef)(nil), "tensorflow.FunctionDef")
	proto.RegisterMapType((map[uint32]*FunctionDef_ArgAttrs)(nil), "tensorflow.FunctionDef.ArgAttrEntry")
	proto.RegisterMapType((map[string]*AttrValue)(nil), "tensorflow.FunctionDef.AttrEntry")
	proto.RegisterMapType((map[string]string)(nil), "tensorflow.FunctionDef.ControlRetEntry")
	proto.RegisterMapType((map[string]string)(nil), "tensorflow.FunctionDef.RetEntry")
	proto.RegisterType((*FunctionDef_ArgAttrs)(nil), "tensorflow.FunctionDef.ArgAttrs")
	proto.RegisterMapType((map[string]*AttrValue)(nil), "tensorflow.FunctionDef.ArgAttrs.AttrEntry")
	proto.RegisterType((*GradientDef)(nil), "tensorflow.GradientDef")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/function.proto", fileDescriptor_507748d6812c5f14)
}

var fileDescriptor_507748d6812c5f14 = []byte{
	// 517 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xac, 0x94, 0xdf, 0x8a, 0xd3, 0x40,
	0x14, 0xc6, 0x49, 0xd3, 0x6d, 0xd3, 0x93, 0xae, 0xae, 0xa3, 0x62, 0xe8, 0x55, 0xad, 0xa2, 0x65,
	0x85, 0x04, 0xba, 0xb8, 0x88, 0xb0, 0x8a, 0xeb, 0x5f, 0x44, 0xea, 0x92, 0x0b, 0x05, 0x11, 0xc2,
	0x34, 0x9d, 0xc4, 0xb0, 0xed, 0x4c, 0x99, 0x4c, 0x5d, 0x7a, 0xe3, 0x83, 0xf8, 0x0c, 0x3e, 0xa0,
	0x97, 0x32, 0x93, 0x4c, 0x32, 0xd6, 0x8d, 0x45, 0xf0, 0x6e, 0x3a, 0xf3, 0xfd, 0xbe, 0x73, 0xfa,
	0x4d, 0xe6, 0xc0, 0x58, 0x10, 0x9a, 0x33, 0x9e, 0x2c, 0xd8, 0x45, 0x10, 0x33, 0x4e, 0x82, 0x84,
	0xe3, 0x25, 0xb9, 0x60, 0xfc, 0x3c, 0x48, 0xd6, 0x34, 0x16, 0x19, 0xa3, 0xfe, 0x8a, 0x33, 0xc1,
	0x10, 0xd4, 0xca, 0xc1, 0x61, 0x33, 0x85, 0x85, 0xe0, 0xd1, 0x57, 0xbc, 0x58, 0x93, 0x82, 0x1b,
	0xfc, 0xa5, 0x02, 0x65, 0x73, 0x12, 0xcd, 0x49, 0x52, 0x2a, 0xef, 0x35, 0x2b, 0xd9, 0xaa, 0xd6,
	0x8d, 0xbe, 0x01, 0x7a, 0x55, 0xf6, 0xf6, 0x82, 0x24, 0xef, 0xb2, 0x19, 0xc7, 0x7c, 0x83, 0x8e,
	0xc0, 0xd1, 0x1d, 0x7b, 0xd6, 0xd0, 0x1e, 0xbb, 0x93, 0x5b, 0x7e, 0x6d, 0xe8, 0x1b, 0x44, 0x58,
	0x09, 0x25, 0x94, 0x72, 0x3c, 0xcf, 0x08, 0x15, 0x5e, 0xeb, 0x4f, 0xe8, 0x75, 0x79, 0xa6, 0x20,
	0x2d, 0x1c, 0xfd, 0xe8, 0x80, 0x6b, 0xd8, 0xa1, 0x00, 0x7a, 0x79, 0x96, 0x52, 0x2c, 0xd6, 0x9c,
	0x78, 0xd6, 0xd0, 0x1a, 0xbb, 0x93, 0x6b, 0xa6, 0xcb, 0xfb, 0x95, 0xe4, 0x6b, 0x0d, 0x7a, 0x08,
	0x6d, 0x19, 0x93, 0xb7, 0xa7, 0x2a, 0xde, 0x6e, 0x68, 0xd3, 0x7f, 0x26, 0x04, 0x7f, 0x49, 0x05,
	0xdf, 0x84, 0x4a, 0x8e, 0x9e, 0x82, 0x83, 0x79, 0x1a, 0x29, 0xb4, 0xab, 0xd0, 0xbb, 0x8d, 0x28,
	0x4f, 0x6b, 0xba, 0x8b, 0x8b, 0x5f, 0xc8, 0x07, 0x47, 0x47, 0xee, 0xd9, 0xca, 0xe0, 0xba, 0x69,
	0x30, 0x65, 0x73, 0x22, 0x3b, 0xed, 0xd2, 0x62, 0x81, 0x26, 0x60, 0x73, 0x22, 0xbc, 0xb6, 0x92,
	0x0e, 0x9b, 0x6a, 0x85, 0x44, 0x14, 0x75, 0xa4, 0x18, 0xbd, 0x01, 0x37, 0x66, 0x54, 0x70, 0xb6,
	0x88, 0x24, 0xdb, 0x51, 0xec, 0xfd, 0x26, 0xf6, 0x79, 0x21, 0xad, 0x2c, 0x20, 0xae, 0x36, 0x06,
	0x53, 0xe8, 0x55, 0xff, 0x01, 0x1d, 0x80, 0x7d, 0x4e, 0x36, 0x2a, 0xdd, 0x5e, 0x28, 0x97, 0xe8,
	0x01, 0xec, 0xa9, 0xcf, 0xcc, 0x6b, 0xa9, 0xc4, 0x6f, 0x9a, 0x25, 0x24, 0xf7, 0x41, 0x1e, 0x86,
	0x85, 0xe6, 0x71, 0xeb, 0x91, 0x35, 0xf8, 0x6e, 0x81, 0x53, 0xe6, 0x92, 0xa3, 0x27, 0xe5, 0x15,
	0x14, 0x5f, 0xca, 0xe1, 0x8e, 0x1c, 0xf3, 0xed, 0xbb, 0xf8, 0xef, 0xcd, 0x7d, 0x86, 0xbe, 0x79,
	0x67, 0xa6, 0xe5, 0x7e, 0x61, 0x79, 0xfc, 0xbb, 0xe5, 0x70, 0x57, 0xcb, 0xa6, 0xfb, 0x31, 0x38,
	0x3a, 0xe2, 0x4b, 0x9a, 0xbd, 0x61, 0x3a, 0xf7, 0x4c, 0xee, 0x04, 0xae, 0x6e, 0xdd, 0xd0, 0xbf,
	0xe0, 0x6f, 0xdb, 0x4e, 0xeb, 0xc0, 0x1e, 0x7d, 0x04, 0xd7, 0x78, 0x47, 0xe8, 0x0e, 0xec, 0xeb,
	0xe7, 0x17, 0x51, 0xbc, 0x24, 0xa5, 0x55, 0x5f, 0x6f, 0x4e, 0xf1, 0x92, 0x48, 0x91, 0x7e, 0x6e,
	0x91, 0x3c, 0x28, 0xbd, 0xfb, 0x7a, 0x53, 0xfe, 0xe1, 0x53, 0x0a, 0x1e, 0xe3, 0xa9, 0x99, 0x43,
	0x35, 0x30, 0x4e, 0xaf, 0xe8, 0x48, 0xce, 0xe4, 0xc8, 0xc8, 0xcf, 0xac, 0x4f, 0x27, 0x69, 0x26,
	0xbe, 0xac, 0x67, 0x7e, 0xcc, 0x96, 0x81, 0x31, 0x68, 0x2e, 0x5f, 0xa6, 0x6c, 0x6b, 0x02, 0xfd,
	0xb4, 0xac, 0x59, 0x47, 0x8d, 0x9f, 0xa3, 0x5f, 0x01, 0x00, 0x00, 0xff, 0xff, 0xab, 0xce, 0xca,
	0x73, 0x34, 0x05, 0x00, 0x00,
}
