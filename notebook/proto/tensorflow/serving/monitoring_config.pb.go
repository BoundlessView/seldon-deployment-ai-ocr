// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow_serving/config/monitoring_config.proto

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

// Configuration for Prometheus monitoring.
type PrometheusConfig struct {
	// Whether to expose Prometheus metrics.
	Enable bool `protobuf:"varint,1,opt,name=enable,proto3" json:"enable,omitempty"`
	// The endpoint to expose Prometheus metrics.
	// If not specified, PrometheusExporter::kPrometheusPath value is used.
	Path                 string   `protobuf:"bytes,2,opt,name=path,proto3" json:"path,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *PrometheusConfig) Reset()         { *m = PrometheusConfig{} }
func (m *PrometheusConfig) String() string { return proto.CompactTextString(m) }
func (*PrometheusConfig) ProtoMessage()    {}
func (*PrometheusConfig) Descriptor() ([]byte, []int) {
	return fileDescriptor_36ed4c52c00502cb, []int{0}
}

func (m *PrometheusConfig) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_PrometheusConfig.Unmarshal(m, b)
}
func (m *PrometheusConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_PrometheusConfig.Marshal(b, m, deterministic)
}
func (m *PrometheusConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_PrometheusConfig.Merge(m, src)
}
func (m *PrometheusConfig) XXX_Size() int {
	return xxx_messageInfo_PrometheusConfig.Size(m)
}
func (m *PrometheusConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_PrometheusConfig.DiscardUnknown(m)
}

var xxx_messageInfo_PrometheusConfig proto.InternalMessageInfo

func (m *PrometheusConfig) GetEnable() bool {
	if m != nil {
		return m.Enable
	}
	return false
}

func (m *PrometheusConfig) GetPath() string {
	if m != nil {
		return m.Path
	}
	return ""
}

// Configuration for monitoring.
type MonitoringConfig struct {
	PrometheusConfig     *PrometheusConfig `protobuf:"bytes,1,opt,name=prometheus_config,json=prometheusConfig,proto3" json:"prometheus_config,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *MonitoringConfig) Reset()         { *m = MonitoringConfig{} }
func (m *MonitoringConfig) String() string { return proto.CompactTextString(m) }
func (*MonitoringConfig) ProtoMessage()    {}
func (*MonitoringConfig) Descriptor() ([]byte, []int) {
	return fileDescriptor_36ed4c52c00502cb, []int{1}
}

func (m *MonitoringConfig) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_MonitoringConfig.Unmarshal(m, b)
}
func (m *MonitoringConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_MonitoringConfig.Marshal(b, m, deterministic)
}
func (m *MonitoringConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MonitoringConfig.Merge(m, src)
}
func (m *MonitoringConfig) XXX_Size() int {
	return xxx_messageInfo_MonitoringConfig.Size(m)
}
func (m *MonitoringConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_MonitoringConfig.DiscardUnknown(m)
}

var xxx_messageInfo_MonitoringConfig proto.InternalMessageInfo

func (m *MonitoringConfig) GetPrometheusConfig() *PrometheusConfig {
	if m != nil {
		return m.PrometheusConfig
	}
	return nil
}

func init() {
	proto.RegisterType((*PrometheusConfig)(nil), "tensorflow.serving.PrometheusConfig")
	proto.RegisterType((*MonitoringConfig)(nil), "tensorflow.serving.MonitoringConfig")
}

func init() {
	proto.RegisterFile("tensorflow_serving/config/monitoring_config.proto", fileDescriptor_36ed4c52c00502cb)
}

var fileDescriptor_36ed4c52c00502cb = []byte{
	// 175 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x32, 0x2c, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0x8f, 0x2f, 0x4e, 0x2d, 0x2a, 0xcb, 0xcc, 0x4b, 0xd7, 0x4f,
	0xce, 0xcf, 0x4b, 0xcb, 0x4c, 0xd7, 0xcf, 0xcd, 0xcf, 0xcb, 0x2c, 0xc9, 0x2f, 0xca, 0xcc, 0x4b,
	0x8f, 0x87, 0x88, 0xe8, 0x15, 0x14, 0xe5, 0x97, 0xe4, 0x0b, 0x09, 0x21, 0xb4, 0xe8, 0x41, 0xb5,
	0x28, 0xd9, 0x71, 0x09, 0x04, 0x14, 0xe5, 0xe7, 0xa6, 0x96, 0x64, 0xa4, 0x96, 0x16, 0x3b, 0x83,
	0x55, 0x0b, 0x89, 0x71, 0xb1, 0xa5, 0xe6, 0x25, 0x26, 0xe5, 0xa4, 0x4a, 0x30, 0x2a, 0x30, 0x6a,
	0x70, 0x04, 0x41, 0x79, 0x42, 0x42, 0x5c, 0x2c, 0x05, 0x89, 0x25, 0x19, 0x12, 0x4c, 0x0a, 0x8c,
	0x1a, 0x9c, 0x41, 0x60, 0xb6, 0x52, 0x2a, 0x97, 0x80, 0x2f, 0xdc, 0x3a, 0xa8, 0xfe, 0x40, 0x2e,
	0xc1, 0x02, 0xb8, 0x99, 0x50, 0x27, 0x80, 0x8d, 0xe2, 0x36, 0x52, 0xd1, 0xc3, 0x74, 0x83, 0x1e,
	0xba, 0x03, 0x82, 0x04, 0x0a, 0xd0, 0x44, 0x9c, 0x98, 0x7f, 0x30, 0x32, 0x26, 0xb1, 0x81, 0xbd,
	0x61, 0x0c, 0x08, 0x00, 0x00, 0xff, 0xff, 0xc2, 0x1d, 0x3a, 0xb9, 0xfb, 0x00, 0x00, 0x00,
}
