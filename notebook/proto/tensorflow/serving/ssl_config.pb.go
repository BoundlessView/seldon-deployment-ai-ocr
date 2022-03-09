// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow_serving/config/ssl_config.proto

package serving

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

// Configuration for a secure gRPC channel
type SSLConfig struct {
	// private server key for SSL
	ServerKey string `protobuf:"bytes,1,opt,name=server_key,json=serverKey,proto3" json:"server_key,omitempty"`
	// public server certificate
	ServerCert string `protobuf:"bytes,2,opt,name=server_cert,json=serverCert,proto3" json:"server_cert,omitempty"`
	//  custom certificate authority
	CustomCa string `protobuf:"bytes,3,opt,name=custom_ca,json=customCa,proto3" json:"custom_ca,omitempty"`
	// valid client certificate required ?
	ClientVerify         bool     `protobuf:"varint,4,opt,name=client_verify,json=clientVerify,proto3" json:"client_verify,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *SSLConfig) Reset()         { *m = SSLConfig{} }
func (m *SSLConfig) String() string { return proto.CompactTextString(m) }
func (*SSLConfig) ProtoMessage()    {}
func (*SSLConfig) Descriptor() ([]byte, []int) {
	return fileDescriptor_4ff8a518289eb8df, []int{0}
}

func (m *SSLConfig) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_SSLConfig.Unmarshal(m, b)
}
func (m *SSLConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_SSLConfig.Marshal(b, m, deterministic)
}
func (m *SSLConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SSLConfig.Merge(m, src)
}
func (m *SSLConfig) XXX_Size() int {
	return xxx_messageInfo_SSLConfig.Size(m)
}
func (m *SSLConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_SSLConfig.DiscardUnknown(m)
}

var xxx_messageInfo_SSLConfig proto.InternalMessageInfo

func (m *SSLConfig) GetServerKey() string {
	if m != nil {
		return m.ServerKey
	}
	return ""
}

func (m *SSLConfig) GetServerCert() string {
	if m != nil {
		return m.ServerCert
	}
	return ""
}

func (m *SSLConfig) GetCustomCa() string {
	if m != nil {
		return m.CustomCa
	}
	return ""
}

func (m *SSLConfig) GetClientVerify() bool {
	if m != nil {
		return m.ClientVerify
	}
	return false
}

func init() {
	proto.RegisterType((*SSLConfig)(nil), "tensorflow.serving.SSLConfig")
}

func init() {
	proto.RegisterFile("tensorflow_serving/config/ssl_config.proto", fileDescriptor_4ff8a518289eb8df)
}

var fileDescriptor_4ff8a518289eb8df = []byte{
	// 187 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x44, 0x8e, 0xb1, 0x0a, 0xc2, 0x30,
	0x14, 0x45, 0x89, 0x15, 0x69, 0xa2, 0x2e, 0x99, 0x02, 0x22, 0x16, 0x5d, 0x8a, 0x43, 0x3b, 0xf8,
	0x07, 0x76, 0xd4, 0xa9, 0x05, 0xd7, 0x50, 0xc3, 0x6b, 0x09, 0xd6, 0x44, 0x92, 0x58, 0xe9, 0x47,
	0xf8, 0xbf, 0x8e, 0x62, 0x52, 0x70, 0x7b, 0x9c, 0x73, 0x1e, 0x5c, 0xb2, 0x77, 0xa0, 0xac, 0x36,
	0x4d, 0xa7, 0x5f, 0xdc, 0x82, 0xe9, 0xa5, 0x6a, 0x73, 0xa1, 0x55, 0x23, 0xdb, 0xdc, 0xda, 0x8e,
	0x87, 0x33, 0x7b, 0x18, 0xed, 0x34, 0xa5, 0xff, 0x36, 0x1b, 0xdb, 0xed, 0x1b, 0x11, 0x5c, 0x55,
	0xe7, 0xc2, 0x77, 0x74, 0x4d, 0xc8, 0x4f, 0x80, 0xe1, 0x37, 0x18, 0x18, 0x4a, 0x50, 0x8a, 0x4b,
	0x1c, 0xc8, 0x09, 0x06, 0xba, 0x21, 0xf3, 0x51, 0x0b, 0x30, 0x8e, 0x4d, 0xbc, 0x1f, 0x3f, 0x0a,
	0x30, 0x8e, 0xae, 0x08, 0x16, 0x4f, 0xeb, 0xf4, 0x9d, 0x8b, 0x9a, 0x45, 0x5e, 0xc7, 0x01, 0x14,
	0x35, 0xdd, 0x91, 0xa5, 0xe8, 0x24, 0x28, 0xc7, 0x7b, 0x30, 0xb2, 0x19, 0xd8, 0x34, 0x41, 0x69,
	0x5c, 0x2e, 0x02, 0xbc, 0x78, 0x76, 0x8c, 0x3e, 0x08, 0x5d, 0x67, 0x7e, 0xef, 0xe1, 0x1b, 0x00,
	0x00, 0xff, 0xff, 0xbc, 0x39, 0x10, 0x99, 0xdd, 0x00, 0x00, 0x00,
}
