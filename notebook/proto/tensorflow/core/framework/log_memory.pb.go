// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/framework/log_memory.proto

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

type MemoryLogStep struct {
	// Process-unique step id.
	StepId int64 `protobuf:"varint,1,opt,name=step_id,json=stepId,proto3" json:"step_id,omitempty"`
	// Handle describing the feeds and fetches of the step.
	Handle               string   `protobuf:"bytes,2,opt,name=handle,proto3" json:"handle,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *MemoryLogStep) Reset()         { *m = MemoryLogStep{} }
func (m *MemoryLogStep) String() string { return proto.CompactTextString(m) }
func (*MemoryLogStep) ProtoMessage()    {}
func (*MemoryLogStep) Descriptor() ([]byte, []int) {
	return fileDescriptor_4f52e83a3ef81427, []int{0}
}

func (m *MemoryLogStep) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MemoryLogStep.Unmarshal(m, b)
}
func (m *MemoryLogStep) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MemoryLogStep.Marshal(b, m, deterministic)
}
func (m *MemoryLogStep) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MemoryLogStep.Merge(m, src)
}
func (m *MemoryLogStep) XXX_Size() int {
	return xxx_messageInfo_MemoryLogStep.Size(m)
}
func (m *MemoryLogStep) XXX_DiscardUnknown() {
	xxx_messageInfo_MemoryLogStep.DiscardUnknown(m)
}

var xxx_messageInfo_MemoryLogStep proto.InternalMessageInfo

func (m *MemoryLogStep) GetStepId() int64 {
	if m != nil {
		return m.StepId
	}
	return 0
}

func (m *MemoryLogStep) GetHandle() string {
	if m != nil {
		return m.Handle
	}
	return ""
}

type MemoryLogTensorAllocation struct {
	// Process-unique step id.
	StepId int64 `protobuf:"varint,1,opt,name=step_id,json=stepId,proto3" json:"step_id,omitempty"`
	// Name of the kernel making the allocation as set in GraphDef,
	// e.g., "affine2/weights/Assign".
	KernelName string `protobuf:"bytes,2,opt,name=kernel_name,json=kernelName,proto3" json:"kernel_name,omitempty"`
	// Allocated tensor details.
	Tensor               *TensorDescription `protobuf:"bytes,3,opt,name=tensor,proto3" json:"tensor,omitempty"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *MemoryLogTensorAllocation) Reset()         { *m = MemoryLogTensorAllocation{} }
func (m *MemoryLogTensorAllocation) String() string { return proto.CompactTextString(m) }
func (*MemoryLogTensorAllocation) ProtoMessage()    {}
func (*MemoryLogTensorAllocation) Descriptor() ([]byte, []int) {
	return fileDescriptor_4f52e83a3ef81427, []int{1}
}

func (m *MemoryLogTensorAllocation) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MemoryLogTensorAllocation.Unmarshal(m, b)
}
func (m *MemoryLogTensorAllocation) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MemoryLogTensorAllocation.Marshal(b, m, deterministic)
}
func (m *MemoryLogTensorAllocation) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MemoryLogTensorAllocation.Merge(m, src)
}
func (m *MemoryLogTensorAllocation) XXX_Size() int {
	return xxx_messageInfo_MemoryLogTensorAllocation.Size(m)
}
func (m *MemoryLogTensorAllocation) XXX_DiscardUnknown() {
	xxx_messageInfo_MemoryLogTensorAllocation.DiscardUnknown(m)
}

var xxx_messageInfo_MemoryLogTensorAllocation proto.InternalMessageInfo

func (m *MemoryLogTensorAllocation) GetStepId() int64 {
	if m != nil {
		return m.StepId
	}
	return 0
}

func (m *MemoryLogTensorAllocation) GetKernelName() string {
	if m != nil {
		return m.KernelName
	}
	return ""
}

func (m *MemoryLogTensorAllocation) GetTensor() *TensorDescription {
	if m != nil {
		return m.Tensor
	}
	return nil
}

type MemoryLogTensorDeallocation struct {
	// Id of the tensor buffer being deallocated, used to match to a
	// corresponding allocation.
	AllocationId int64 `protobuf:"varint,1,opt,name=allocation_id,json=allocationId,proto3" json:"allocation_id,omitempty"`
	// Name of the allocator used.
	AllocatorName        string   `protobuf:"bytes,2,opt,name=allocator_name,json=allocatorName,proto3" json:"allocator_name,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *MemoryLogTensorDeallocation) Reset()         { *m = MemoryLogTensorDeallocation{} }
func (m *MemoryLogTensorDeallocation) String() string { return proto.CompactTextString(m) }
func (*MemoryLogTensorDeallocation) ProtoMessage()    {}
func (*MemoryLogTensorDeallocation) Descriptor() ([]byte, []int) {
	return fileDescriptor_4f52e83a3ef81427, []int{2}
}

func (m *MemoryLogTensorDeallocation) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MemoryLogTensorDeallocation.Unmarshal(m, b)
}
func (m *MemoryLogTensorDeallocation) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MemoryLogTensorDeallocation.Marshal(b, m, deterministic)
}
func (m *MemoryLogTensorDeallocation) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MemoryLogTensorDeallocation.Merge(m, src)
}
func (m *MemoryLogTensorDeallocation) XXX_Size() int {
	return xxx_messageInfo_MemoryLogTensorDeallocation.Size(m)
}
func (m *MemoryLogTensorDeallocation) XXX_DiscardUnknown() {
	xxx_messageInfo_MemoryLogTensorDeallocation.DiscardUnknown(m)
}

var xxx_messageInfo_MemoryLogTensorDeallocation proto.InternalMessageInfo

func (m *MemoryLogTensorDeallocation) GetAllocationId() int64 {
	if m != nil {
		return m.AllocationId
	}
	return 0
}

func (m *MemoryLogTensorDeallocation) GetAllocatorName() string {
	if m != nil {
		return m.AllocatorName
	}
	return ""
}

type MemoryLogTensorOutput struct {
	// Process-unique step id.
	StepId int64 `protobuf:"varint,1,opt,name=step_id,json=stepId,proto3" json:"step_id,omitempty"`
	// Name of the kernel producing an output as set in GraphDef, e.g.,
	// "affine2/weights/Assign".
	KernelName string `protobuf:"bytes,2,opt,name=kernel_name,json=kernelName,proto3" json:"kernel_name,omitempty"`
	// Index of the output being set.
	Index int32 `protobuf:"varint,3,opt,name=index,proto3" json:"index,omitempty"`
	// Output tensor details.
	Tensor               *TensorDescription `protobuf:"bytes,4,opt,name=tensor,proto3" json:"tensor,omitempty"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *MemoryLogTensorOutput) Reset()         { *m = MemoryLogTensorOutput{} }
func (m *MemoryLogTensorOutput) String() string { return proto.CompactTextString(m) }
func (*MemoryLogTensorOutput) ProtoMessage()    {}
func (*MemoryLogTensorOutput) Descriptor() ([]byte, []int) {
	return fileDescriptor_4f52e83a3ef81427, []int{3}
}

func (m *MemoryLogTensorOutput) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MemoryLogTensorOutput.Unmarshal(m, b)
}
func (m *MemoryLogTensorOutput) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MemoryLogTensorOutput.Marshal(b, m, deterministic)
}
func (m *MemoryLogTensorOutput) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MemoryLogTensorOutput.Merge(m, src)
}
func (m *MemoryLogTensorOutput) XXX_Size() int {
	return xxx_messageInfo_MemoryLogTensorOutput.Size(m)
}
func (m *MemoryLogTensorOutput) XXX_DiscardUnknown() {
	xxx_messageInfo_MemoryLogTensorOutput.DiscardUnknown(m)
}

var xxx_messageInfo_MemoryLogTensorOutput proto.InternalMessageInfo

func (m *MemoryLogTensorOutput) GetStepId() int64 {
	if m != nil {
		return m.StepId
	}
	return 0
}

func (m *MemoryLogTensorOutput) GetKernelName() string {
	if m != nil {
		return m.KernelName
	}
	return ""
}

func (m *MemoryLogTensorOutput) GetIndex() int32 {
	if m != nil {
		return m.Index
	}
	return 0
}

func (m *MemoryLogTensorOutput) GetTensor() *TensorDescription {
	if m != nil {
		return m.Tensor
	}
	return nil
}

type MemoryLogRawAllocation struct {
	// Process-unique step id.
	StepId int64 `protobuf:"varint,1,opt,name=step_id,json=stepId,proto3" json:"step_id,omitempty"`
	// Name of the operation making the allocation.
	Operation string `protobuf:"bytes,2,opt,name=operation,proto3" json:"operation,omitempty"`
	// Number of bytes in the allocation.
	NumBytes int64 `protobuf:"varint,3,opt,name=num_bytes,json=numBytes,proto3" json:"num_bytes,omitempty"`
	// Address of the allocation.
	Ptr uint64 `protobuf:"varint,4,opt,name=ptr,proto3" json:"ptr,omitempty"`
	// Id of the tensor buffer being allocated, used to match to a
	// corresponding deallocation.
	AllocationId int64 `protobuf:"varint,5,opt,name=allocation_id,json=allocationId,proto3" json:"allocation_id,omitempty"`
	// Name of the allocator used.
	AllocatorName        string   `protobuf:"bytes,6,opt,name=allocator_name,json=allocatorName,proto3" json:"allocator_name,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *MemoryLogRawAllocation) Reset()         { *m = MemoryLogRawAllocation{} }
func (m *MemoryLogRawAllocation) String() string { return proto.CompactTextString(m) }
func (*MemoryLogRawAllocation) ProtoMessage()    {}
func (*MemoryLogRawAllocation) Descriptor() ([]byte, []int) {
	return fileDescriptor_4f52e83a3ef81427, []int{4}
}

func (m *MemoryLogRawAllocation) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MemoryLogRawAllocation.Unmarshal(m, b)
}
func (m *MemoryLogRawAllocation) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MemoryLogRawAllocation.Marshal(b, m, deterministic)
}
func (m *MemoryLogRawAllocation) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MemoryLogRawAllocation.Merge(m, src)
}
func (m *MemoryLogRawAllocation) XXX_Size() int {
	return xxx_messageInfo_MemoryLogRawAllocation.Size(m)
}
func (m *MemoryLogRawAllocation) XXX_DiscardUnknown() {
	xxx_messageInfo_MemoryLogRawAllocation.DiscardUnknown(m)
}

var xxx_messageInfo_MemoryLogRawAllocation proto.InternalMessageInfo

func (m *MemoryLogRawAllocation) GetStepId() int64 {
	if m != nil {
		return m.StepId
	}
	return 0
}

func (m *MemoryLogRawAllocation) GetOperation() string {
	if m != nil {
		return m.Operation
	}
	return ""
}

func (m *MemoryLogRawAllocation) GetNumBytes() int64 {
	if m != nil {
		return m.NumBytes
	}
	return 0
}

func (m *MemoryLogRawAllocation) GetPtr() uint64 {
	if m != nil {
		return m.Ptr
	}
	return 0
}

func (m *MemoryLogRawAllocation) GetAllocationId() int64 {
	if m != nil {
		return m.AllocationId
	}
	return 0
}

func (m *MemoryLogRawAllocation) GetAllocatorName() string {
	if m != nil {
		return m.AllocatorName
	}
	return ""
}

type MemoryLogRawDeallocation struct {
	// Process-unique step id.
	StepId int64 `protobuf:"varint,1,opt,name=step_id,json=stepId,proto3" json:"step_id,omitempty"`
	// Name of the operation making the deallocation.
	Operation string `protobuf:"bytes,2,opt,name=operation,proto3" json:"operation,omitempty"`
	// Id of the tensor buffer being deallocated, used to match to a
	// corresponding allocation.
	AllocationId int64 `protobuf:"varint,3,opt,name=allocation_id,json=allocationId,proto3" json:"allocation_id,omitempty"`
	// Name of the allocator used.
	AllocatorName string `protobuf:"bytes,4,opt,name=allocator_name,json=allocatorName,proto3" json:"allocator_name,omitempty"`
	// True if the deallocation is queued and will be performed later,
	// e.g. for GPU lazy freeing of buffers.
	Deferred             bool     `protobuf:"varint,5,opt,name=deferred,proto3" json:"deferred,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *MemoryLogRawDeallocation) Reset()         { *m = MemoryLogRawDeallocation{} }
func (m *MemoryLogRawDeallocation) String() string { return proto.CompactTextString(m) }
func (*MemoryLogRawDeallocation) ProtoMessage()    {}
func (*MemoryLogRawDeallocation) Descriptor() ([]byte, []int) {
	return fileDescriptor_4f52e83a3ef81427, []int{5}
}

func (m *MemoryLogRawDeallocation) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MemoryLogRawDeallocation.Unmarshal(m, b)
}
func (m *MemoryLogRawDeallocation) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MemoryLogRawDeallocation.Marshal(b, m, deterministic)
}
func (m *MemoryLogRawDeallocation) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MemoryLogRawDeallocation.Merge(m, src)
}
func (m *MemoryLogRawDeallocation) XXX_Size() int {
	return xxx_messageInfo_MemoryLogRawDeallocation.Size(m)
}
func (m *MemoryLogRawDeallocation) XXX_DiscardUnknown() {
	xxx_messageInfo_MemoryLogRawDeallocation.DiscardUnknown(m)
}

var xxx_messageInfo_MemoryLogRawDeallocation proto.InternalMessageInfo

func (m *MemoryLogRawDeallocation) GetStepId() int64 {
	if m != nil {
		return m.StepId
	}
	return 0
}

func (m *MemoryLogRawDeallocation) GetOperation() string {
	if m != nil {
		return m.Operation
	}
	return ""
}

func (m *MemoryLogRawDeallocation) GetAllocationId() int64 {
	if m != nil {
		return m.AllocationId
	}
	return 0
}

func (m *MemoryLogRawDeallocation) GetAllocatorName() string {
	if m != nil {
		return m.AllocatorName
	}
	return ""
}

func (m *MemoryLogRawDeallocation) GetDeferred() bool {
	if m != nil {
		return m.Deferred
	}
	return false
}

func init() {
	proto.RegisterType((*MemoryLogStep)(nil), "tensorflow.MemoryLogStep")
	proto.RegisterType((*MemoryLogTensorAllocation)(nil), "tensorflow.MemoryLogTensorAllocation")
	proto.RegisterType((*MemoryLogTensorDeallocation)(nil), "tensorflow.MemoryLogTensorDeallocation")
	proto.RegisterType((*MemoryLogTensorOutput)(nil), "tensorflow.MemoryLogTensorOutput")
	proto.RegisterType((*MemoryLogRawAllocation)(nil), "tensorflow.MemoryLogRawAllocation")
	proto.RegisterType((*MemoryLogRawDeallocation)(nil), "tensorflow.MemoryLogRawDeallocation")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/log_memory.proto", fileDescriptor_4f52e83a3ef81427)
}

var fileDescriptor_4f52e83a3ef81427 = []byte{
	// 441 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xa4, 0x93, 0xc1, 0x6e, 0xd3, 0x40,
	0x10, 0x86, 0xb5, 0x38, 0x31, 0xc9, 0x94, 0x00, 0x5a, 0x41, 0x31, 0x2d, 0x88, 0x28, 0x08, 0x29,
	0xe2, 0x90, 0x48, 0x45, 0x1c, 0x91, 0x20, 0xea, 0xa5, 0x52, 0x81, 0x6a, 0xe1, 0xc4, 0xc5, 0x72,
	0xe2, 0x89, 0x6b, 0xd5, 0xbb, 0x63, 0xad, 0xd7, 0x0a, 0x7d, 0x07, 0x9e, 0x81, 0xf7, 0xe0, 0x15,
	0x78, 0x22, 0x8e, 0xc8, 0xde, 0xe0, 0x35, 0x69, 0x2a, 0x05, 0x7a, 0xf3, 0x3f, 0x9e, 0x9d, 0xf9,
	0xfe, 0x19, 0x0d, 0xbc, 0x34, 0xa8, 0x0a, 0xd2, 0xcb, 0x8c, 0x56, 0xd3, 0x05, 0x69, 0x9c, 0x2e,
	0x75, 0x24, 0x71, 0x45, 0xfa, 0x62, 0x9a, 0x51, 0x12, 0x4a, 0x94, 0xa4, 0x2f, 0x27, 0xb9, 0x26,
	0x43, 0x1c, 0x5c, 0xee, 0xc1, 0xd1, 0xf5, 0xef, 0xec, 0x9f, 0x30, 0xc6, 0x62, 0xa1, 0xd3, 0xdc,
	0xa4, 0xa4, 0xec, 0xfb, 0xd1, 0x5b, 0x18, 0xbc, 0xaf, 0xeb, 0x9d, 0x52, 0xf2, 0xc9, 0x60, 0xce,
	0x1f, 0xc1, 0xed, 0xc2, 0x60, 0x1e, 0xa6, 0x71, 0xc0, 0x86, 0x6c, 0xec, 0x09, 0xbf, 0x92, 0x27,
	0x31, 0xdf, 0x07, 0xff, 0x3c, 0x52, 0x71, 0x86, 0xc1, 0xad, 0x21, 0x1b, 0xf7, 0xc5, 0x5a, 0x8d,
	0xbe, 0x31, 0x78, 0xdc, 0x94, 0xf8, 0x5c, 0xf7, 0x79, 0x97, 0x65, 0xb4, 0x88, 0xaa, 0x2e, 0xd7,
	0x97, 0x7b, 0x06, 0x7b, 0x17, 0xa8, 0x15, 0x66, 0xa1, 0x8a, 0xe4, 0x9f, 0x9a, 0x60, 0x43, 0x1f,
	0x22, 0x89, 0xfc, 0x35, 0xf8, 0x96, 0x3a, 0xf0, 0x86, 0x6c, 0xbc, 0x77, 0xf4, 0x74, 0xe2, 0xec,
	0x4d, 0x6c, 0x9f, 0x63, 0x67, 0x47, 0xac, 0x93, 0x47, 0x29, 0x1c, 0x6e, 0xd0, 0x1c, 0x63, 0xe4,
	0x78, 0x9e, 0xc3, 0xc0, 0x29, 0x47, 0x75, 0xc7, 0x05, 0x4f, 0x62, 0xfe, 0x02, 0xee, 0xae, 0x35,
	0xe9, 0x36, 0xde, 0xa0, 0x89, 0x56, 0x84, 0xa3, 0xef, 0x0c, 0x1e, 0x6e, 0xf4, 0xfa, 0x58, 0x9a,
	0xbc, 0x34, 0x37, 0x70, 0xfd, 0x00, 0xba, 0xa9, 0x8a, 0xf1, 0x6b, 0x6d, 0xba, 0x2b, 0xac, 0x68,
	0xcd, 0xa2, 0xf3, 0x2f, 0xb3, 0xf8, 0xc9, 0x60, 0xbf, 0x01, 0x14, 0xd1, 0x6a, 0x97, 0xbd, 0x3c,
	0x81, 0x3e, 0xe5, 0xa8, 0xeb, 0xac, 0x35, 0x9f, 0x0b, 0xf0, 0x43, 0xe8, 0xab, 0x52, 0x86, 0xf3,
	0x4b, 0x83, 0x45, 0x8d, 0xe8, 0x89, 0x9e, 0x2a, 0xe5, 0xac, 0xd2, 0xfc, 0x3e, 0x78, 0xb9, 0xb1,
	0x88, 0x1d, 0x51, 0x7d, 0x5e, 0x9d, 0x76, 0x77, 0xa7, 0x69, 0xfb, 0xdb, 0xa6, 0xfd, 0x83, 0x41,
	0xd0, 0x36, 0xf3, 0xd7, 0x5a, 0xff, 0xd3, 0xce, 0x15, 0x3e, 0x6f, 0x27, 0xbe, 0xce, 0x16, 0x3e,
	0x7e, 0x00, 0xbd, 0x18, 0x97, 0xa8, 0x35, 0x5a, 0x9b, 0x3d, 0xd1, 0xe8, 0x19, 0x41, 0x40, 0x3a,
	0x69, 0x2f, 0xad, 0x39, 0xcd, 0xd9, 0xbd, 0x53, 0x4a, 0xac, 0xaf, 0xb3, 0xea, 0x22, 0x8b, 0x33,
	0xf6, 0xe5, 0x4d, 0x92, 0x9a, 0xf3, 0x72, 0x3e, 0x59, 0x90, 0x9c, 0xb6, 0x6e, 0x7a, 0xfb, 0x67,
	0x42, 0x1b, 0xc7, 0xfe, 0x8b, 0xb1, 0xb9, 0x5f, 0x5f, 0xf7, 0xab, 0xdf, 0x01, 0x00, 0x00, 0xff,
	0xff, 0xfc, 0x6c, 0x8e, 0x46, 0x4b, 0x04, 0x00, 0x00,
}
