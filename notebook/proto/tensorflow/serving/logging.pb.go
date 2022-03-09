// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow_serving/core/logging.proto

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

// Metadata logged along with the request logs.
type LogMetadata struct {
	ModelSpec      *ModelSpec      `protobuf:"bytes,1,opt,name=model_spec,json=modelSpec,proto3" json:"model_spec,omitempty"`
	SamplingConfig *SamplingConfig `protobuf:"bytes,2,opt,name=sampling_config,json=samplingConfig,proto3" json:"sampling_config,omitempty"`
	// List of tags used to load the relevant MetaGraphDef from SavedModel.
	SavedModelTags       []string `protobuf:"bytes,3,rep,name=saved_model_tags,json=savedModelTags,proto3" json:"saved_model_tags,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *LogMetadata) Reset()         { *m = LogMetadata{} }
func (m *LogMetadata) String() string { return proto.CompactTextString(m) }
func (*LogMetadata) ProtoMessage()    {}
func (*LogMetadata) Descriptor() ([]byte, []int) {
	return fileDescriptor_b61adc125a8f9545, []int{0}
}

func (m *LogMetadata) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_LogMetadata.Unmarshal(m, b)
}
func (m *LogMetadata) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_LogMetadata.Marshal(b, m, deterministic)
}
func (m *LogMetadata) XXX_Merge(src proto.Message) {
	xxx_messageInfo_LogMetadata.Merge(m, src)
}
func (m *LogMetadata) XXX_Size() int {
	return xxx_messageInfo_LogMetadata.Size(m)
}
func (m *LogMetadata) XXX_DiscardUnknown() {
	xxx_messageInfo_LogMetadata.DiscardUnknown(m)
}

var xxx_messageInfo_LogMetadata proto.InternalMessageInfo

func (m *LogMetadata) GetModelSpec() *ModelSpec {
	if m != nil {
		return m.ModelSpec
	}
	return nil
}

func (m *LogMetadata) GetSamplingConfig() *SamplingConfig {
	if m != nil {
		return m.SamplingConfig
	}
	return nil
}

func (m *LogMetadata) GetSavedModelTags() []string {
	if m != nil {
		return m.SavedModelTags
	}
	return nil
}

func init() {
	proto.RegisterType((*LogMetadata)(nil), "tensorflow.serving.LogMetadata")
}

func init() {
	proto.RegisterFile("tensorflow_serving/core/logging.proto", fileDescriptor_b61adc125a8f9545)
}

var fileDescriptor_b61adc125a8f9545 = []byte{
	// 225 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x6c, 0x8f, 0x41, 0x4b, 0xc3, 0x40,
	0x10, 0x85, 0x59, 0x03, 0x42, 0xb7, 0xa0, 0xb2, 0xa7, 0x50, 0x10, 0x4a, 0x45, 0xc8, 0x69, 0x03,
	0x7a, 0xf5, 0xa4, 0x47, 0xed, 0xa5, 0xf5, 0x1e, 0xc6, 0x64, 0x3a, 0x04, 0x92, 0xcc, 0xb2, 0xb3,
	0xc4, 0xff, 0xe7, 0xaf, 0xf2, 0x28, 0xdd, 0x6d, 0x09, 0xd2, 0xdc, 0x76, 0xe0, 0x7b, 0xdf, 0x7b,
	0xab, 0x1f, 0x03, 0x0e, 0xc2, 0xfe, 0xd0, 0xf1, 0x77, 0x25, 0xe8, 0xc7, 0x76, 0xa0, 0xb2, 0x66,
	0x8f, 0x65, 0xc7, 0x44, 0xed, 0x40, 0xd6, 0x79, 0x0e, 0x6c, 0xcc, 0x84, 0xd9, 0x13, 0xb6, 0x7a,
	0x98, 0x89, 0x82, 0x6b, 0xa5, 0xec, 0xb9, 0xc1, 0x2e, 0x05, 0x57, 0x76, 0xd6, 0x3f, 0x1c, 0x5a,
	0x3a, 0x37, 0x54, 0xe9, 0x4c, 0xfc, 0xe6, 0x47, 0xe9, 0xe5, 0x07, 0xd3, 0x16, 0x03, 0x34, 0x10,
	0xc0, 0xbc, 0x68, 0x1d, 0x75, 0x95, 0x38, 0xac, 0x73, 0xb5, 0x56, 0xc5, 0xf2, 0xe9, 0xde, 0x5e,
	0xae, 0xb1, 0xdb, 0x23, 0xb5, 0x77, 0x58, 0xef, 0x16, 0xfd, 0xf9, 0x69, 0xde, 0xf5, 0xad, 0x40,
	0xef, 0xba, 0xa9, 0x26, 0xbf, 0x8a, 0x8a, 0xcd, 0x9c, 0x62, 0x7f, 0x42, 0xdf, 0x22, 0xb9, 0xbb,
	0x91, 0x7f, 0xb7, 0x29, 0xf4, 0x9d, 0xc0, 0x88, 0x4d, 0x95, 0x06, 0x05, 0x20, 0xc9, 0xb3, 0x75,
	0x56, 0x2c, 0x8e, 0xe4, 0x88, 0x4d, 0x5c, 0xf0, 0x09, 0x24, 0xaf, 0xd9, 0xaf, 0x52, 0x5f, 0xd7,
	0xf1, 0x43, 0xcf, 0x7f, 0x01, 0x00, 0x00, 0xff, 0xff, 0xad, 0x15, 0x8a, 0xde, 0x62, 0x01, 0x00,
	0x00,
}
