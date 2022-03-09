// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/protobuf/debug.proto

package protobuf

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

// Option for watching a node in TensorFlow Debugger (tfdbg).
type DebugTensorWatch struct {
	// Name of the node to watch.
	// Use "*" for wildcard. But note: currently, regex is not supported in
	// general.
	NodeName string `protobuf:"bytes,1,opt,name=node_name,json=nodeName,proto3" json:"node_name,omitempty"`
	// Output slot to watch.
	// The semantics of output_slot == -1 is that all outputs of the node
	// will be watched (i.e., a wildcard).
	// Other negative values of output_slot are invalid and will lead to
	// errors currently.
	OutputSlot int32 `protobuf:"varint,2,opt,name=output_slot,json=outputSlot,proto3" json:"output_slot,omitempty"`
	// Name(s) of the debugging op(s).
	// One or more than one probes on a tensor.
	// e.g., {"DebugIdentity", "DebugNanCount"}
	DebugOps []string `protobuf:"bytes,3,rep,name=debug_ops,json=debugOps,proto3" json:"debug_ops,omitempty"`
	// URL(s) for debug targets(s).
	//
	// Supported URL formats are:
	//   - file:///foo/tfdbg_dump: Writes out Event content to file
	//     /foo/tfdbg_dump.  Assumes all directories can be created if they don't
	//     already exist.
	//   - grpc://localhost:11011: Sends an RPC request to an EventListener
	//     service running at localhost:11011 with the event.
	//   - memcbk:///event_key: Routes tensors to clients using the
	//     callback registered with the DebugCallbackRegistry for event_key.
	//
	// Each debug op listed in debug_ops will publish its output tensor (debug
	// signal) to all URLs in debug_urls.
	//
	// N.B. Session::Run() supports concurrent invocations of the same inputs
	// (feed keys), outputs and target nodes. If such concurrent invocations
	// are to be debugged, the callers of Session::Run() must use distinct
	// debug_urls to make sure that the streamed or dumped events do not overlap
	// among the invocations.
	// TODO(cais): More visible documentation of this in g3docs.
	DebugUrls []string `protobuf:"bytes,4,rep,name=debug_urls,json=debugUrls,proto3" json:"debug_urls,omitempty"`
	// Do not error out if debug op creation fails (e.g., due to dtype
	// incompatibility). Instead, just log the failure.
	TolerateDebugOpCreationFailures bool     `protobuf:"varint,5,opt,name=tolerate_debug_op_creation_failures,json=tolerateDebugOpCreationFailures,proto3" json:"tolerate_debug_op_creation_failures,omitempty"`
	XXX_NoUnkeyedLiteral            struct{} `json:"-"`
	XXX_unrecognized                []byte   `json:"-"`
	XXX_sizecache                   int32    `json:"-"`
}

func (m *DebugTensorWatch) Reset()         { *m = DebugTensorWatch{} }
func (m *DebugTensorWatch) String() string { return proto.CompactTextString(m) }
func (*DebugTensorWatch) ProtoMessage()    {}
func (*DebugTensorWatch) Descriptor() ([]byte, []int) {
	return fileDescriptor_4fbf764b7c91eef6, []int{0}
}

func (m *DebugTensorWatch) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_DebugTensorWatch.Unmarshal(m, b)
}
func (m *DebugTensorWatch) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_DebugTensorWatch.Marshal(b, m, deterministic)
}
func (m *DebugTensorWatch) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DebugTensorWatch.Merge(m, src)
}
func (m *DebugTensorWatch) XXX_Size() int {
	return xxx_messageInfo_DebugTensorWatch.Size(m)
}
func (m *DebugTensorWatch) XXX_DiscardUnknown() {
	xxx_messageInfo_DebugTensorWatch.DiscardUnknown(m)
}

var xxx_messageInfo_DebugTensorWatch proto.InternalMessageInfo

func (m *DebugTensorWatch) GetNodeName() string {
	if m != nil {
		return m.NodeName
	}
	return ""
}

func (m *DebugTensorWatch) GetOutputSlot() int32 {
	if m != nil {
		return m.OutputSlot
	}
	return 0
}

func (m *DebugTensorWatch) GetDebugOps() []string {
	if m != nil {
		return m.DebugOps
	}
	return nil
}

func (m *DebugTensorWatch) GetDebugUrls() []string {
	if m != nil {
		return m.DebugUrls
	}
	return nil
}

func (m *DebugTensorWatch) GetTolerateDebugOpCreationFailures() bool {
	if m != nil {
		return m.TolerateDebugOpCreationFailures
	}
	return false
}

// Options for initializing DebuggerState in TensorFlow Debugger (tfdbg).
type DebugOptions struct {
	// Debugging options
	DebugTensorWatchOpts []*DebugTensorWatch `protobuf:"bytes,4,rep,name=debug_tensor_watch_opts,json=debugTensorWatchOpts,proto3" json:"debug_tensor_watch_opts,omitempty"`
	// Caller-specified global step count.
	// Note that this is distinct from the session run count and the executor
	// step count.
	GlobalStep int64 `protobuf:"varint,10,opt,name=global_step,json=globalStep,proto3" json:"global_step,omitempty"`
	// Whether the total disk usage of tfdbg is to be reset to zero
	// in this Session.run call. This is used by wrappers and hooks
	// such as the local CLI ones to indicate that the dumped tensors
	// are cleaned up from the disk after each Session.run.
	ResetDiskByteUsage   bool     `protobuf:"varint,11,opt,name=reset_disk_byte_usage,json=resetDiskByteUsage,proto3" json:"reset_disk_byte_usage,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *DebugOptions) Reset()         { *m = DebugOptions{} }
func (m *DebugOptions) String() string { return proto.CompactTextString(m) }
func (*DebugOptions) ProtoMessage()    {}
func (*DebugOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_4fbf764b7c91eef6, []int{1}
}

func (m *DebugOptions) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_DebugOptions.Unmarshal(m, b)
}
func (m *DebugOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_DebugOptions.Marshal(b, m, deterministic)
}
func (m *DebugOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DebugOptions.Merge(m, src)
}
func (m *DebugOptions) XXX_Size() int {
	return xxx_messageInfo_DebugOptions.Size(m)
}
func (m *DebugOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_DebugOptions.DiscardUnknown(m)
}

var xxx_messageInfo_DebugOptions proto.InternalMessageInfo

func (m *DebugOptions) GetDebugTensorWatchOpts() []*DebugTensorWatch {
	if m != nil {
		return m.DebugTensorWatchOpts
	}
	return nil
}

func (m *DebugOptions) GetGlobalStep() int64 {
	if m != nil {
		return m.GlobalStep
	}
	return 0
}

func (m *DebugOptions) GetResetDiskByteUsage() bool {
	if m != nil {
		return m.ResetDiskByteUsage
	}
	return false
}

type DebuggedSourceFile struct {
	// The host name on which a source code file is located.
	Host string `protobuf:"bytes,1,opt,name=host,proto3" json:"host,omitempty"`
	// Path to the source code file.
	FilePath string `protobuf:"bytes,2,opt,name=file_path,json=filePath,proto3" json:"file_path,omitempty"`
	// The timestamp at which the source code file is last modified.
	LastModified int64 `protobuf:"varint,3,opt,name=last_modified,json=lastModified,proto3" json:"last_modified,omitempty"`
	// Byte size of the file.
	Bytes int64 `protobuf:"varint,4,opt,name=bytes,proto3" json:"bytes,omitempty"`
	// Line-by-line content of the source code file.
	Lines                []string `protobuf:"bytes,5,rep,name=lines,proto3" json:"lines,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *DebuggedSourceFile) Reset()         { *m = DebuggedSourceFile{} }
func (m *DebuggedSourceFile) String() string { return proto.CompactTextString(m) }
func (*DebuggedSourceFile) ProtoMessage()    {}
func (*DebuggedSourceFile) Descriptor() ([]byte, []int) {
	return fileDescriptor_4fbf764b7c91eef6, []int{2}
}

func (m *DebuggedSourceFile) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_DebuggedSourceFile.Unmarshal(m, b)
}
func (m *DebuggedSourceFile) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_DebuggedSourceFile.Marshal(b, m, deterministic)
}
func (m *DebuggedSourceFile) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DebuggedSourceFile.Merge(m, src)
}
func (m *DebuggedSourceFile) XXX_Size() int {
	return xxx_messageInfo_DebuggedSourceFile.Size(m)
}
func (m *DebuggedSourceFile) XXX_DiscardUnknown() {
	xxx_messageInfo_DebuggedSourceFile.DiscardUnknown(m)
}

var xxx_messageInfo_DebuggedSourceFile proto.InternalMessageInfo

func (m *DebuggedSourceFile) GetHost() string {
	if m != nil {
		return m.Host
	}
	return ""
}

func (m *DebuggedSourceFile) GetFilePath() string {
	if m != nil {
		return m.FilePath
	}
	return ""
}

func (m *DebuggedSourceFile) GetLastModified() int64 {
	if m != nil {
		return m.LastModified
	}
	return 0
}

func (m *DebuggedSourceFile) GetBytes() int64 {
	if m != nil {
		return m.Bytes
	}
	return 0
}

func (m *DebuggedSourceFile) GetLines() []string {
	if m != nil {
		return m.Lines
	}
	return nil
}

type DebuggedSourceFiles struct {
	// A collection of source code files.
	SourceFiles          []*DebuggedSourceFile `protobuf:"bytes,1,rep,name=source_files,json=sourceFiles,proto3" json:"source_files,omitempty"`
	XXX_NoUnkeyedLiteral struct{}              `json:"-"`
	XXX_unrecognized     []byte                `json:"-"`
	XXX_sizecache        int32                 `json:"-"`
}

func (m *DebuggedSourceFiles) Reset()         { *m = DebuggedSourceFiles{} }
func (m *DebuggedSourceFiles) String() string { return proto.CompactTextString(m) }
func (*DebuggedSourceFiles) ProtoMessage()    {}
func (*DebuggedSourceFiles) Descriptor() ([]byte, []int) {
	return fileDescriptor_4fbf764b7c91eef6, []int{3}
}

func (m *DebuggedSourceFiles) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_DebuggedSourceFiles.Unmarshal(m, b)
}
func (m *DebuggedSourceFiles) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_DebuggedSourceFiles.Marshal(b, m, deterministic)
}
func (m *DebuggedSourceFiles) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DebuggedSourceFiles.Merge(m, src)
}
func (m *DebuggedSourceFiles) XXX_Size() int {
	return xxx_messageInfo_DebuggedSourceFiles.Size(m)
}
func (m *DebuggedSourceFiles) XXX_DiscardUnknown() {
	xxx_messageInfo_DebuggedSourceFiles.DiscardUnknown(m)
}

var xxx_messageInfo_DebuggedSourceFiles proto.InternalMessageInfo

func (m *DebuggedSourceFiles) GetSourceFiles() []*DebuggedSourceFile {
	if m != nil {
		return m.SourceFiles
	}
	return nil
}

func init() {
	proto.RegisterType((*DebugTensorWatch)(nil), "tensorflow.DebugTensorWatch")
	proto.RegisterType((*DebugOptions)(nil), "tensorflow.DebugOptions")
	proto.RegisterType((*DebuggedSourceFile)(nil), "tensorflow.DebuggedSourceFile")
	proto.RegisterType((*DebuggedSourceFiles)(nil), "tensorflow.DebuggedSourceFiles")
}

func init() {
	proto.RegisterFile("tensorflow/core/protobuf/debug.proto", fileDescriptor_4fbf764b7c91eef6)
}

var fileDescriptor_4fbf764b7c91eef6 = []byte{
	// 483 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x6c, 0x93, 0xcd, 0x6e, 0xd3, 0x40,
	0x10, 0xc7, 0xb5, 0xa4, 0x45, 0xed, 0x24, 0x48, 0x68, 0x29, 0xc2, 0x12, 0x1f, 0x8d, 0x52, 0x0e,
	0x39, 0x25, 0x02, 0xae, 0x5c, 0x08, 0x51, 0x4f, 0x40, 0x23, 0x87, 0x0a, 0xc4, 0x65, 0xb5, 0x8e,
	0xc7, 0x8e, 0xc9, 0x26, 0x63, 0xed, 0x8c, 0x15, 0xf5, 0x45, 0x78, 0x15, 0xde, 0x85, 0x27, 0xe1,
	0x88, 0x76, 0x9d, 0x2a, 0xfd, 0xe0, 0xe6, 0xf9, 0xfd, 0x67, 0x66, 0x67, 0xff, 0xb3, 0x86, 0xd7,
	0x82, 0x1b, 0x26, 0x5f, 0x38, 0xda, 0x8e, 0x17, 0xe4, 0x71, 0x5c, 0x7b, 0x12, 0xca, 0x9a, 0x62,
	0x9c, 0x63, 0xd6, 0x94, 0xa3, 0x18, 0x6a, 0xd8, 0x67, 0x0d, 0xfe, 0x28, 0x78, 0x3c, 0x0d, 0xda,
	0xd7, 0xc8, 0xbe, 0x59, 0x59, 0x2c, 0xf5, 0x73, 0x38, 0xde, 0x50, 0x8e, 0x66, 0x63, 0xd7, 0x98,
	0xa8, 0xbe, 0x1a, 0x1e, 0xa7, 0x47, 0x01, 0x7c, 0xb1, 0x6b, 0xd4, 0xa7, 0xd0, 0xa5, 0x46, 0xea,
	0x46, 0x0c, 0x3b, 0x92, 0xe4, 0x41, 0x5f, 0x0d, 0x0f, 0x53, 0x68, 0xd1, 0xdc, 0x91, 0x84, 0xea,
	0x78, 0x9a, 0xa1, 0x9a, 0x93, 0x4e, 0xbf, 0x13, 0xaa, 0x23, 0xb8, 0xa8, 0x59, 0xbf, 0x04, 0x68,
	0xc5, 0xc6, 0x3b, 0x4e, 0x0e, 0xa2, 0xda, 0xa6, 0x5f, 0x7a, 0xc7, 0xfa, 0x13, 0x9c, 0x09, 0x39,
	0xf4, 0x56, 0xd0, 0x5c, 0x37, 0x31, 0x0b, 0x8f, 0x56, 0x2a, 0xda, 0x98, 0xc2, 0x56, 0xae, 0xf1,
	0xc8, 0xc9, 0x61, 0x5f, 0x0d, 0x8f, 0xd2, 0xd3, 0xeb, 0xd4, 0x69, 0xdb, 0xfd, 0xe3, 0x2e, 0xef,
	0x7c, 0x97, 0x36, 0xf8, 0xad, 0xa0, 0xb7, 0xd3, 0x02, 0x67, 0x3d, 0x87, 0x67, 0x6d, 0xd7, 0xd6,
	0x01, 0xb3, 0x0d, 0xd7, 0x35, 0x54, 0x4b, 0x3b, 0x4a, 0xf7, 0xed, 0x8b, 0xd1, 0xde, 0x9b, 0xd1,
	0x5d, 0x5f, 0xd2, 0x93, 0xfc, 0x0e, 0xb9, 0xa8, 0x85, 0x83, 0x21, 0xa5, 0xa3, 0xcc, 0x3a, 0xc3,
	0x82, 0x75, 0x02, 0x7d, 0x35, 0xec, 0xa4, 0xd0, 0xa2, 0xb9, 0x60, 0xad, 0xdf, 0xc0, 0x53, 0x8f,
	0x8c, 0x62, 0xf2, 0x8a, 0x57, 0x26, 0xbb, 0x12, 0x34, 0x0d, 0xdb, 0x12, 0x93, 0x6e, 0xbc, 0x86,
	0x8e, 0xe2, 0xb4, 0xe2, 0xd5, 0xe4, 0x4a, 0xf0, 0x32, 0x28, 0x83, 0x5f, 0x0a, 0x74, 0x3c, 0xbe,
	0xc4, 0x7c, 0x4e, 0x8d, 0x5f, 0xe0, 0x79, 0xe5, 0x50, 0x6b, 0x38, 0x58, 0x12, 0xcb, 0x6e, 0x27,
	0xf1, 0x3b, 0xd8, 0x5d, 0x54, 0x0e, 0x4d, 0x6d, 0x65, 0x19, 0xb7, 0x71, 0x9c, 0x1e, 0x05, 0x30,
	0xb3, 0xb2, 0xd4, 0x67, 0xf0, 0xc8, 0x59, 0x16, 0xb3, 0xa6, 0xbc, 0x2a, 0x2a, 0xcc, 0x93, 0x4e,
	0x9c, 0xae, 0x17, 0xe0, 0xe7, 0x1d, 0xd3, 0x27, 0x70, 0x18, 0x86, 0x0a, 0x1e, 0x04, 0xb1, 0x0d,
	0x02, 0x75, 0xd5, 0x26, 0x9a, 0x1d, 0x96, 0xd4, 0x06, 0x83, 0xef, 0xf0, 0xe4, 0xfe, 0x5c, 0xac,
	0x3f, 0x40, 0x8f, 0x63, 0x68, 0xc2, 0xd1, 0x9c, 0xa8, 0xe8, 0xe6, 0xab, 0x7b, 0x6e, 0xde, 0x2a,
	0x4b, 0xbb, 0xbc, 0x6f, 0x31, 0xf9, 0x09, 0x09, 0xf9, 0xf2, 0x66, 0x45, 0xe1, 0xed, 0x1a, 0xb7,
	0xe4, 0x57, 0x93, 0x6e, 0x2c, 0x9e, 0x85, 0xd7, 0xcb, 0x33, 0xf5, 0xe3, 0x7d, 0x59, 0xc9, 0xb2,
	0xc9, 0x46, 0x0b, 0x5a, 0x8f, 0x6f, 0xbc, 0xf8, 0xff, 0x7f, 0x96, 0x74, 0xfb, 0x57, 0xf8, 0xab,
	0x54, 0xf6, 0x30, 0x06, 0xef, 0xfe, 0x05, 0x00, 0x00, 0xff, 0xff, 0x8a, 0x4e, 0xc2, 0xa9, 0x30,
	0x03, 0x00, 0x00,
}
