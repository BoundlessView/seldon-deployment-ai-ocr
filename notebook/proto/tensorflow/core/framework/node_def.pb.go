// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/framework/node_def.proto

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

type NodeDef struct {
	// The name given to this operator. Used for naming inputs,
	// logging, visualization, etc.  Unique within a single GraphDef.
	// Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_>./]*".
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// The operation name.  There may be custom parameters in attrs.
	// Op names starting with an underscore are reserved for internal use.
	Op string `protobuf:"bytes,2,opt,name=op,proto3" json:"op,omitempty"`
	// Each input is "node:src_output" with "node" being a string name and
	// "src_output" indicating which output tensor to use from "node". If
	// "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
	// may optionally be followed by control inputs that have the format
	// "^node".
	Input []string `protobuf:"bytes,3,rep,name=input,proto3" json:"input,omitempty"`
	// A (possibly partial) specification for the device on which this
	// node should be placed.
	// The expected syntax for this string is as follows:
	//
	// DEVICE_SPEC ::= PARTIAL_SPEC
	//
	// PARTIAL_SPEC ::= ("/" CONSTRAINT) *
	// CONSTRAINT ::= ("job:" JOB_NAME)
	//              | ("replica:" [1-9][0-9]*)
	//              | ("task:" [1-9][0-9]*)
	//              | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*") )
	//
	// Valid values for this string include:
	// * "/job:worker/replica:0/task:1/device:GPU:3"  (full specification)
	// * "/job:worker/device:GPU:3"                   (partial specification)
	// * ""                                    (no specification)
	//
	// If the constraints do not resolve to a single device (or if this
	// field is empty or not present), the runtime will attempt to
	// choose a device automatically.
	Device string `protobuf:"bytes,4,opt,name=device,proto3" json:"device,omitempty"`
	// Operation-specific graph-construction-time configuration.
	// Note that this should include all attrs defined in the
	// corresponding OpDef, including those with a value matching
	// the default -- this allows the default to change and makes
	// NodeDefs easier to interpret on their own.  However, if
	// an attr with a default is not specified in this list, the
	// default will be used.
	// The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
	// one of the names from the corresponding OpDef's attr field).
	// The values must have a type matching the corresponding OpDef
	// attr's type field.
	// TODO(josh11b): Add some examples here showing best practices.
	Attr map[string]*AttrValue `protobuf:"bytes,5,rep,name=attr,proto3" json:"attr,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	// This stores debug information associated with the node.
	ExperimentalDebugInfo *NodeDef_ExperimentalDebugInfo `protobuf:"bytes,6,opt,name=experimental_debug_info,json=experimentalDebugInfo,proto3" json:"experimental_debug_info,omitempty"`
	XXX_NoUnkeyedLiteral  struct{}                       `json:"-"`
	XXX_unrecognized      []byte                         `json:"-"`
	XXX_sizecache         int32                          `json:"-"`
}

func (m *NodeDef) Reset()         { *m = NodeDef{} }
func (m *NodeDef) String() string { return proto.CompactTextString(m) }
func (*NodeDef) ProtoMessage()    {}
func (*NodeDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_b34b3b836a96140b, []int{0}
}

func (m *NodeDef) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_NodeDef.Unmarshal(m, b)
}
func (m *NodeDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_NodeDef.Marshal(b, m, deterministic)
}
func (m *NodeDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NodeDef.Merge(m, src)
}
func (m *NodeDef) XXX_Size() int {
	return xxx_messageInfo_NodeDef.Size(m)
}
func (m *NodeDef) XXX_DiscardUnknown() {
	xxx_messageInfo_NodeDef.DiscardUnknown(m)
}

var xxx_messageInfo_NodeDef proto.InternalMessageInfo

func (m *NodeDef) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *NodeDef) GetOp() string {
	if m != nil {
		return m.Op
	}
	return ""
}

func (m *NodeDef) GetInput() []string {
	if m != nil {
		return m.Input
	}
	return nil
}

func (m *NodeDef) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *NodeDef) GetAttr() map[string]*AttrValue {
	if m != nil {
		return m.Attr
	}
	return nil
}

func (m *NodeDef) GetExperimentalDebugInfo() *NodeDef_ExperimentalDebugInfo {
	if m != nil {
		return m.ExperimentalDebugInfo
	}
	return nil
}

type NodeDef_ExperimentalDebugInfo struct {
	// Opaque string inserted into error messages created by the runtime.
	//
	// This is intended to store the list of names of the nodes from the
	// original graph that this node was derived. For example if this node, say
	// C, was result of a fusion of 2 nodes A and B, then 'original_node' would
	// be {A, B}. This information can be used to map errors originating at the
	// current node to some top level source code.
	OriginalNodeNames []string `protobuf:"bytes,1,rep,name=original_node_names,json=originalNodeNames,proto3" json:"original_node_names,omitempty"`
	// This is intended to store the list of names of the functions from the
	// original graph that this node was derived. For example if this node, say
	// C, was result of a fusion of node A in function FA and node B in function
	// FB, then `original_funcs` would be {FA, FB}. If the node is in the top
	// level graph, the `original_func` is empty. This information, with the
	// `original_node_names` can be used to map errors originating at the
	// current ndoe to some top level source code.
	OriginalFuncNames    []string `protobuf:"bytes,2,rep,name=original_func_names,json=originalFuncNames,proto3" json:"original_func_names,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *NodeDef_ExperimentalDebugInfo) Reset()         { *m = NodeDef_ExperimentalDebugInfo{} }
func (m *NodeDef_ExperimentalDebugInfo) String() string { return proto.CompactTextString(m) }
func (*NodeDef_ExperimentalDebugInfo) ProtoMessage()    {}
func (*NodeDef_ExperimentalDebugInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_b34b3b836a96140b, []int{0, 1}
}

func (m *NodeDef_ExperimentalDebugInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_NodeDef_ExperimentalDebugInfo.Unmarshal(m, b)
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_NodeDef_ExperimentalDebugInfo.Marshal(b, m, deterministic)
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NodeDef_ExperimentalDebugInfo.Merge(m, src)
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_Size() int {
	return xxx_messageInfo_NodeDef_ExperimentalDebugInfo.Size(m)
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_NodeDef_ExperimentalDebugInfo.DiscardUnknown(m)
}

var xxx_messageInfo_NodeDef_ExperimentalDebugInfo proto.InternalMessageInfo

func (m *NodeDef_ExperimentalDebugInfo) GetOriginalNodeNames() []string {
	if m != nil {
		return m.OriginalNodeNames
	}
	return nil
}

func (m *NodeDef_ExperimentalDebugInfo) GetOriginalFuncNames() []string {
	if m != nil {
		return m.OriginalFuncNames
	}
	return nil
}

func init() {
	proto.RegisterType((*NodeDef)(nil), "tensorflow.NodeDef")
	proto.RegisterMapType((map[string]*AttrValue)(nil), "tensorflow.NodeDef.AttrEntry")
	proto.RegisterType((*NodeDef_ExperimentalDebugInfo)(nil), "tensorflow.NodeDef.ExperimentalDebugInfo")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/node_def.proto", fileDescriptor_b34b3b836a96140b)
}

var fileDescriptor_b34b3b836a96140b = []byte{
	// 374 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x92, 0x4f, 0xeb, 0xd3, 0x30,
	0x18, 0xc7, 0x49, 0xbb, 0x4d, 0x9a, 0x81, 0x68, 0x74, 0x1a, 0x06, 0x42, 0xf1, 0x54, 0x15, 0x5a,
	0x9c, 0x17, 0x11, 0x3c, 0x38, 0x36, 0xc1, 0xcb, 0x18, 0x3d, 0x78, 0xf0, 0x52, 0xba, 0xf6, 0x69,
	0x0d, 0x6b, 0xf3, 0x94, 0x2c, 0xdd, 0xdc, 0x5b, 0xf3, 0x95, 0x79, 0x94, 0x64, 0x75, 0xeb, 0xa4,
	0xbf, 0xdb, 0x93, 0x3c, 0x9f, 0x6f, 0xbe, 0x79, 0xfe, 0xd0, 0x40, 0x83, 0x3c, 0xa0, 0x2a, 0x2a,
	0x3c, 0x45, 0x19, 0x2a, 0x88, 0x0a, 0x95, 0xd6, 0x70, 0x42, 0xb5, 0x8f, 0x24, 0xe6, 0x90, 0xe4,
	0x50, 0x84, 0x8d, 0x42, 0x8d, 0x8c, 0xde, 0xc8, 0xf9, 0xdb, 0x87, 0x55, 0xa9, 0xd6, 0x2a, 0x39,
	0xa6, 0x55, 0x0b, 0x17, 0xdd, 0xeb, 0xdf, 0x2e, 0x7d, 0xb4, 0xc1, 0x1c, 0x56, 0x50, 0x30, 0x46,
	0x47, 0x32, 0xad, 0x81, 0x13, 0x9f, 0x04, 0x5e, 0x6c, 0x63, 0xf6, 0x98, 0x3a, 0xd8, 0x70, 0xc7,
	0xde, 0x38, 0xd8, 0xb0, 0xe7, 0x74, 0x2c, 0x64, 0xd3, 0x6a, 0xee, 0xfa, 0x6e, 0xe0, 0xc5, 0x97,
	0x03, 0x7b, 0x41, 0x27, 0x39, 0x1c, 0x45, 0x06, 0x7c, 0x64, 0xc9, 0xee, 0xc4, 0xde, 0xd3, 0x91,
	0x71, 0xe4, 0x63, 0xdf, 0x0d, 0xa6, 0x8b, 0x57, 0xe1, 0xed, 0x63, 0x61, 0x67, 0x1a, 0x7e, 0xd1,
	0x5a, 0xad, 0xa5, 0x56, 0xe7, 0xd8, 0xa2, 0x2c, 0xa5, 0x2f, 0xe1, 0x57, 0x03, 0x4a, 0xd4, 0x20,
	0x75, 0x5a, 0x25, 0x39, 0xec, 0xda, 0x32, 0x11, 0xb2, 0x40, 0x3e, 0xf1, 0x49, 0x30, 0x5d, 0xbc,
	0x19, 0x7a, 0x65, 0xdd, 0x93, 0xac, 0x8c, 0xe2, 0x9b, 0x2c, 0x30, 0x9e, 0xc1, 0xd0, 0xf5, 0x7c,
	0x43, 0xbd, 0xab, 0x2b, 0x7b, 0x42, 0xdd, 0x3d, 0x9c, 0xbb, 0x9a, 0x4d, 0xc8, 0xde, 0xd1, 0xb1,
	0xed, 0x90, 0xad, 0x7a, 0xba, 0x98, 0xf5, 0xfd, 0x8c, 0xee, 0xbb, 0x49, 0xc6, 0x17, 0xe6, 0x93,
	0xf3, 0x91, 0xcc, 0x4f, 0x74, 0x36, 0xe8, 0xcf, 0x42, 0xfa, 0x0c, 0x95, 0x28, 0x85, 0x4c, 0xab,
	0xc4, 0xce, 0xcb, 0xb4, 0xf4, 0xc0, 0x89, 0x6d, 0xdd, 0xd3, 0x7f, 0x29, 0x53, 0xc3, 0xc6, 0x24,
	0xee, 0xf8, 0xa2, 0x95, 0x59, 0xc7, 0x3b, 0xf7, 0xfc, 0xd7, 0x56, 0x66, 0x96, 0x5f, 0x0a, 0xca,
	0x51, 0x95, 0xfd, 0xff, 0x5d, 0x27, 0xbd, 0xf4, 0xcc, 0xb3, 0x5b, 0x33, 0xe3, 0x2d, 0xf9, 0xf1,
	0xb9, 0x14, 0xfa, 0x67, 0xbb, 0x0b, 0x33, 0xac, 0xa3, 0xde, 0x72, 0x0c, 0x87, 0x25, 0xfe, 0xb7,
	0x35, 0x7f, 0x08, 0xd9, 0x4d, 0xec, 0xba, 0x7c, 0xf8, 0x1b, 0x00, 0x00, 0xff, 0xff, 0xb5, 0xf2,
	0xb4, 0x7b, 0x92, 0x02, 0x00, 0x00,
}
