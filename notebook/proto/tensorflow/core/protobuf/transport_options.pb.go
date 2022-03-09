// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/protobuf/transport_options.proto

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

// Extra data needed on a non-RDMA RecvBufResponse.
type RecvBufRespExtra struct {
	TensorContent        [][]byte `protobuf:"bytes,1,rep,name=tensor_content,json=tensorContent,proto3" json:"tensor_content,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *RecvBufRespExtra) Reset()         { *m = RecvBufRespExtra{} }
func (m *RecvBufRespExtra) String() string { return proto.CompactTextString(m) }
func (*RecvBufRespExtra) ProtoMessage()    {}
func (*RecvBufRespExtra) Descriptor() ([]byte, []int) {
	return fileDescriptor_527891df7bab7653, []int{0}
}

func (m *RecvBufRespExtra) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RecvBufRespExtra.Unmarshal(m, b)
}
func (m *RecvBufRespExtra) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RecvBufRespExtra.Marshal(b, m, deterministic)
}
func (m *RecvBufRespExtra) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RecvBufRespExtra.Merge(m, src)
}
func (m *RecvBufRespExtra) XXX_Size() int {
	return xxx_messageInfo_RecvBufRespExtra.Size(m)
}
func (m *RecvBufRespExtra) XXX_DiscardUnknown() {
	xxx_messageInfo_RecvBufRespExtra.DiscardUnknown(m)
}

var xxx_messageInfo_RecvBufRespExtra proto.InternalMessageInfo

func (m *RecvBufRespExtra) GetTensorContent() [][]byte {
	if m != nil {
		return m.TensorContent
	}
	return nil
}

func init() {
	proto.RegisterType((*RecvBufRespExtra)(nil), "tensorflow.RecvBufRespExtra")
}

func init() {
	proto.RegisterFile("tensorflow/core/protobuf/transport_options.proto", fileDescriptor_527891df7bab7653)
}

var fileDescriptor_527891df7bab7653 = []byte{
	// 127 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x32, 0x28, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0xd7, 0x4f, 0xce, 0x2f, 0x4a, 0xd5, 0x2f, 0x28, 0xca, 0x2f,
	0xc9, 0x4f, 0x2a, 0x4d, 0xd3, 0x2f, 0x29, 0x4a, 0xcc, 0x2b, 0x2e, 0xc8, 0x2f, 0x2a, 0x89, 0xcf,
	0x2f, 0x28, 0xc9, 0xcc, 0xcf, 0x2b, 0xd6, 0x03, 0x4b, 0x09, 0x71, 0x21, 0x74, 0x28, 0x59, 0x72,
	0x09, 0x04, 0xa5, 0x26, 0x97, 0x39, 0x95, 0xa6, 0x05, 0xa5, 0x16, 0x17, 0xb8, 0x56, 0x94, 0x14,
	0x25, 0x0a, 0xa9, 0x72, 0xf1, 0x41, 0x54, 0xc4, 0x27, 0xe7, 0xe7, 0x95, 0xa4, 0xe6, 0x95, 0x48,
	0x30, 0x2a, 0x30, 0x6b, 0xf0, 0x04, 0xf1, 0x42, 0x44, 0x9d, 0x21, 0x82, 0x49, 0x6c, 0x60, 0xd3,
	0x8c, 0x01, 0x01, 0x00, 0x00, 0xff, 0xff, 0x78, 0x65, 0xc5, 0x6c, 0x81, 0x00, 0x00, 0x00,
}
