// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mi_message.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "mi_message.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace medical_imaging {
class MsgPoint2DefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<MsgPoint2> {
} _MsgPoint2_default_instance_;
class MsgPagingDefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<MsgPaging> {
} _MsgPaging_default_instance_;
class MsgRotateDefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<MsgRotate> {
} _MsgRotate_default_instance_;

namespace protobuf_mi_5fmessage_2eproto {


namespace {

::google::protobuf::Metadata file_level_metadata[3];

}  // namespace

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTableField
    const TableStruct::entries[] = {
  {0, 0, 0, ::google::protobuf::internal::kInvalidMask, 0, 0},
};

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::AuxillaryParseTableField
    const TableStruct::aux[] = {
  ::google::protobuf::internal::AuxillaryParseTableField(),
};
PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTable const
    TableStruct::schema[] = {
  { NULL, NULL, 0, -1, -1, false },
  { NULL, NULL, 0, -1, -1, false },
  { NULL, NULL, 0, -1, -1, false },
};

const ::google::protobuf::uint32 TableStruct::offsets[] = {
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgPoint2, _has_bits_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgPoint2, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgPoint2, x_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgPoint2, y_),
  0,
  1,
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgPaging, _has_bits_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgPaging, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgPaging, page_),
  0,
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgRotate, _has_bits_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgRotate, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgRotate, pre_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MsgRotate, cur_),
  0,
  1,
};

static const ::google::protobuf::internal::MigrationSchema schemas[] = {
  { 0, 7, sizeof(MsgPoint2)},
  { 9, 15, sizeof(MsgPaging)},
  { 16, 23, sizeof(MsgRotate)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&_MsgPoint2_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&_MsgPaging_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&_MsgRotate_default_instance_),
};

namespace {

void protobuf_AssignDescriptors() {
  AddDescriptors();
  ::google::protobuf::MessageFactory* factory = NULL;
  AssignDescriptors(
      "mi_message.proto", schemas, file_default_instances, TableStruct::offsets, factory,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 3);
}

}  // namespace

void TableStruct::Shutdown() {
  _MsgPoint2_default_instance_.Shutdown();
  delete file_level_metadata[0].reflection;
  _MsgPaging_default_instance_.Shutdown();
  delete file_level_metadata[1].reflection;
  _MsgRotate_default_instance_.Shutdown();
  delete file_level_metadata[2].reflection;
}

void TableStruct::InitDefaultsImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::internal::InitProtobufDefaults();
  _MsgPoint2_default_instance_.DefaultConstruct();
  _MsgPaging_default_instance_.DefaultConstruct();
  _MsgRotate_default_instance_.DefaultConstruct();
  _MsgRotate_default_instance_.get_mutable()->pre_ = const_cast< ::medical_imaging::MsgPoint2*>(
      ::medical_imaging::MsgPoint2::internal_default_instance());
  _MsgRotate_default_instance_.get_mutable()->cur_ = const_cast< ::medical_imaging::MsgPoint2*>(
      ::medical_imaging::MsgPoint2::internal_default_instance());
}

void InitDefaults() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &TableStruct::InitDefaultsImpl);
}
void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] = {
      "\n\020mi_message.proto\022\017medical_imaging\"!\n\tM"
      "sgPoint2\022\t\n\001x\030\001 \002(\002\022\t\n\001y\030\002 \002(\002\"\031\n\tMsgPag"
      "ing\022\014\n\004page\030\001 \002(\005\"]\n\tMsgRotate\022\'\n\003pre\030\001 "
      "\002(\0132\032.medical_imaging.MsgPoint2\022\'\n\003cur\030\002"
      " \002(\0132\032.medical_imaging.MsgPoint2"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 192);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "mi_message.proto", &protobuf_RegisterTypes);
  ::google::protobuf::internal::OnShutdown(&TableStruct::Shutdown);
}

void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;

}  // namespace protobuf_mi_5fmessage_2eproto


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int MsgPoint2::kXFieldNumber;
const int MsgPoint2::kYFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

MsgPoint2::MsgPoint2()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_mi_5fmessage_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:medical_imaging.MsgPoint2)
}
MsgPoint2::MsgPoint2(const MsgPoint2& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      _has_bits_(from._has_bits_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&x_, &from.x_,
    reinterpret_cast<char*>(&y_) -
    reinterpret_cast<char*>(&x_) + sizeof(y_));
  // @@protoc_insertion_point(copy_constructor:medical_imaging.MsgPoint2)
}

void MsgPoint2::SharedCtor() {
  _cached_size_ = 0;
  ::memset(&x_, 0, reinterpret_cast<char*>(&y_) -
    reinterpret_cast<char*>(&x_) + sizeof(y_));
}

MsgPoint2::~MsgPoint2() {
  // @@protoc_insertion_point(destructor:medical_imaging.MsgPoint2)
  SharedDtor();
}

void MsgPoint2::SharedDtor() {
}

void MsgPoint2::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* MsgPoint2::descriptor() {
  protobuf_mi_5fmessage_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_mi_5fmessage_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const MsgPoint2& MsgPoint2::default_instance() {
  protobuf_mi_5fmessage_2eproto::InitDefaults();
  return *internal_default_instance();
}

MsgPoint2* MsgPoint2::New(::google::protobuf::Arena* arena) const {
  MsgPoint2* n = new MsgPoint2;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void MsgPoint2::Clear() {
// @@protoc_insertion_point(message_clear_start:medical_imaging.MsgPoint2)
  if (_has_bits_[0 / 32] & 3u) {
    ::memset(&x_, 0, reinterpret_cast<char*>(&y_) -
      reinterpret_cast<char*>(&x_) + sizeof(y_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

bool MsgPoint2::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:medical_imaging.MsgPoint2)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required float x = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(13u)) {
          set_has_x();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &x_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // required float y = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(21u)) {
          set_has_y();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &y_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:medical_imaging.MsgPoint2)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:medical_imaging.MsgPoint2)
  return false;
#undef DO_
}

void MsgPoint2::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:medical_imaging.MsgPoint2)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required float x = 1;
  if (cached_has_bits & 0x00000001u) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(1, this->x(), output);
  }

  // required float y = 2;
  if (cached_has_bits & 0x00000002u) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(2, this->y(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:medical_imaging.MsgPoint2)
}

::google::protobuf::uint8* MsgPoint2::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:medical_imaging.MsgPoint2)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required float x = 1;
  if (cached_has_bits & 0x00000001u) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(1, this->x(), target);
  }

  // required float y = 2;
  if (cached_has_bits & 0x00000002u) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(2, this->y(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:medical_imaging.MsgPoint2)
  return target;
}

size_t MsgPoint2::RequiredFieldsByteSizeFallback() const {
// @@protoc_insertion_point(required_fields_byte_size_fallback_start:medical_imaging.MsgPoint2)
  size_t total_size = 0;

  if (has_x()) {
    // required float x = 1;
    total_size += 1 + 4;
  }

  if (has_y()) {
    // required float y = 2;
    total_size += 1 + 4;
  }

  return total_size;
}
size_t MsgPoint2::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:medical_imaging.MsgPoint2)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  if (((_has_bits_[0] & 0x00000003) ^ 0x00000003) == 0) {  // All required fields are present.
    // required float x = 1;
    total_size += 1 + 4;

    // required float y = 2;
    total_size += 1 + 4;

  } else {
    total_size += RequiredFieldsByteSizeFallback();
  }
  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void MsgPoint2::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:medical_imaging.MsgPoint2)
  GOOGLE_DCHECK_NE(&from, this);
  const MsgPoint2* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const MsgPoint2>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:medical_imaging.MsgPoint2)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:medical_imaging.MsgPoint2)
    MergeFrom(*source);
  }
}

void MsgPoint2::MergeFrom(const MsgPoint2& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:medical_imaging.MsgPoint2)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 3u) {
    if (cached_has_bits & 0x00000001u) {
      x_ = from.x_;
    }
    if (cached_has_bits & 0x00000002u) {
      y_ = from.y_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void MsgPoint2::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:medical_imaging.MsgPoint2)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void MsgPoint2::CopyFrom(const MsgPoint2& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:medical_imaging.MsgPoint2)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MsgPoint2::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000003) != 0x00000003) return false;
  return true;
}

void MsgPoint2::Swap(MsgPoint2* other) {
  if (other == this) return;
  InternalSwap(other);
}
void MsgPoint2::InternalSwap(MsgPoint2* other) {
  std::swap(x_, other->x_);
  std::swap(y_, other->y_);
  std::swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata MsgPoint2::GetMetadata() const {
  protobuf_mi_5fmessage_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_mi_5fmessage_2eproto::file_level_metadata[kIndexInFileMessages];
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// MsgPoint2

// required float x = 1;
bool MsgPoint2::has_x() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
void MsgPoint2::set_has_x() {
  _has_bits_[0] |= 0x00000001u;
}
void MsgPoint2::clear_has_x() {
  _has_bits_[0] &= ~0x00000001u;
}
void MsgPoint2::clear_x() {
  x_ = 0;
  clear_has_x();
}
float MsgPoint2::x() const {
  // @@protoc_insertion_point(field_get:medical_imaging.MsgPoint2.x)
  return x_;
}
void MsgPoint2::set_x(float value) {
  set_has_x();
  x_ = value;
  // @@protoc_insertion_point(field_set:medical_imaging.MsgPoint2.x)
}

// required float y = 2;
bool MsgPoint2::has_y() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
void MsgPoint2::set_has_y() {
  _has_bits_[0] |= 0x00000002u;
}
void MsgPoint2::clear_has_y() {
  _has_bits_[0] &= ~0x00000002u;
}
void MsgPoint2::clear_y() {
  y_ = 0;
  clear_has_y();
}
float MsgPoint2::y() const {
  // @@protoc_insertion_point(field_get:medical_imaging.MsgPoint2.y)
  return y_;
}
void MsgPoint2::set_y(float value) {
  set_has_y();
  y_ = value;
  // @@protoc_insertion_point(field_set:medical_imaging.MsgPoint2.y)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int MsgPaging::kPageFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

MsgPaging::MsgPaging()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_mi_5fmessage_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:medical_imaging.MsgPaging)
}
MsgPaging::MsgPaging(const MsgPaging& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      _has_bits_(from._has_bits_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  page_ = from.page_;
  // @@protoc_insertion_point(copy_constructor:medical_imaging.MsgPaging)
}

void MsgPaging::SharedCtor() {
  _cached_size_ = 0;
  page_ = 0;
}

MsgPaging::~MsgPaging() {
  // @@protoc_insertion_point(destructor:medical_imaging.MsgPaging)
  SharedDtor();
}

void MsgPaging::SharedDtor() {
}

void MsgPaging::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* MsgPaging::descriptor() {
  protobuf_mi_5fmessage_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_mi_5fmessage_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const MsgPaging& MsgPaging::default_instance() {
  protobuf_mi_5fmessage_2eproto::InitDefaults();
  return *internal_default_instance();
}

MsgPaging* MsgPaging::New(::google::protobuf::Arena* arena) const {
  MsgPaging* n = new MsgPaging;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void MsgPaging::Clear() {
// @@protoc_insertion_point(message_clear_start:medical_imaging.MsgPaging)
  page_ = 0;
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

bool MsgPaging::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:medical_imaging.MsgPaging)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required int32 page = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u)) {
          set_has_page();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &page_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:medical_imaging.MsgPaging)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:medical_imaging.MsgPaging)
  return false;
#undef DO_
}

void MsgPaging::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:medical_imaging.MsgPaging)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required int32 page = 1;
  if (cached_has_bits & 0x00000001u) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(1, this->page(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:medical_imaging.MsgPaging)
}

::google::protobuf::uint8* MsgPaging::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:medical_imaging.MsgPaging)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required int32 page = 1;
  if (cached_has_bits & 0x00000001u) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(1, this->page(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:medical_imaging.MsgPaging)
  return target;
}

size_t MsgPaging::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:medical_imaging.MsgPaging)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  // required int32 page = 1;
  if (has_page()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->page());
  }
  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void MsgPaging::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:medical_imaging.MsgPaging)
  GOOGLE_DCHECK_NE(&from, this);
  const MsgPaging* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const MsgPaging>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:medical_imaging.MsgPaging)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:medical_imaging.MsgPaging)
    MergeFrom(*source);
  }
}

void MsgPaging::MergeFrom(const MsgPaging& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:medical_imaging.MsgPaging)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.has_page()) {
    set_page(from.page());
  }
}

void MsgPaging::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:medical_imaging.MsgPaging)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void MsgPaging::CopyFrom(const MsgPaging& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:medical_imaging.MsgPaging)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MsgPaging::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000001) != 0x00000001) return false;
  return true;
}

void MsgPaging::Swap(MsgPaging* other) {
  if (other == this) return;
  InternalSwap(other);
}
void MsgPaging::InternalSwap(MsgPaging* other) {
  std::swap(page_, other->page_);
  std::swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata MsgPaging::GetMetadata() const {
  protobuf_mi_5fmessage_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_mi_5fmessage_2eproto::file_level_metadata[kIndexInFileMessages];
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// MsgPaging

// required int32 page = 1;
bool MsgPaging::has_page() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
void MsgPaging::set_has_page() {
  _has_bits_[0] |= 0x00000001u;
}
void MsgPaging::clear_has_page() {
  _has_bits_[0] &= ~0x00000001u;
}
void MsgPaging::clear_page() {
  page_ = 0;
  clear_has_page();
}
::google::protobuf::int32 MsgPaging::page() const {
  // @@protoc_insertion_point(field_get:medical_imaging.MsgPaging.page)
  return page_;
}
void MsgPaging::set_page(::google::protobuf::int32 value) {
  set_has_page();
  page_ = value;
  // @@protoc_insertion_point(field_set:medical_imaging.MsgPaging.page)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int MsgRotate::kPreFieldNumber;
const int MsgRotate::kCurFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

MsgRotate::MsgRotate()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_mi_5fmessage_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:medical_imaging.MsgRotate)
}
MsgRotate::MsgRotate(const MsgRotate& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      _has_bits_(from._has_bits_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.has_pre()) {
    pre_ = new ::medical_imaging::MsgPoint2(*from.pre_);
  } else {
    pre_ = NULL;
  }
  if (from.has_cur()) {
    cur_ = new ::medical_imaging::MsgPoint2(*from.cur_);
  } else {
    cur_ = NULL;
  }
  // @@protoc_insertion_point(copy_constructor:medical_imaging.MsgRotate)
}

void MsgRotate::SharedCtor() {
  _cached_size_ = 0;
  ::memset(&pre_, 0, reinterpret_cast<char*>(&cur_) -
    reinterpret_cast<char*>(&pre_) + sizeof(cur_));
}

MsgRotate::~MsgRotate() {
  // @@protoc_insertion_point(destructor:medical_imaging.MsgRotate)
  SharedDtor();
}

void MsgRotate::SharedDtor() {
  if (this != internal_default_instance()) {
    delete pre_;
  }
  if (this != internal_default_instance()) {
    delete cur_;
  }
}

void MsgRotate::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* MsgRotate::descriptor() {
  protobuf_mi_5fmessage_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_mi_5fmessage_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const MsgRotate& MsgRotate::default_instance() {
  protobuf_mi_5fmessage_2eproto::InitDefaults();
  return *internal_default_instance();
}

MsgRotate* MsgRotate::New(::google::protobuf::Arena* arena) const {
  MsgRotate* n = new MsgRotate;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void MsgRotate::Clear() {
// @@protoc_insertion_point(message_clear_start:medical_imaging.MsgRotate)
  if (_has_bits_[0 / 32] & 3u) {
    if (has_pre()) {
      GOOGLE_DCHECK(pre_ != NULL);
      pre_->::medical_imaging::MsgPoint2::Clear();
    }
    if (has_cur()) {
      GOOGLE_DCHECK(cur_ != NULL);
      cur_->::medical_imaging::MsgPoint2::Clear();
    }
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

bool MsgRotate::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:medical_imaging.MsgRotate)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required .medical_imaging.MsgPoint2 pre = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_pre()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // required .medical_imaging.MsgPoint2 cur = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_cur()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:medical_imaging.MsgRotate)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:medical_imaging.MsgRotate)
  return false;
#undef DO_
}

void MsgRotate::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:medical_imaging.MsgRotate)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required .medical_imaging.MsgPoint2 pre = 1;
  if (cached_has_bits & 0x00000001u) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, *this->pre_, output);
  }

  // required .medical_imaging.MsgPoint2 cur = 2;
  if (cached_has_bits & 0x00000002u) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, *this->cur_, output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:medical_imaging.MsgRotate)
}

::google::protobuf::uint8* MsgRotate::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:medical_imaging.MsgRotate)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required .medical_imaging.MsgPoint2 pre = 1;
  if (cached_has_bits & 0x00000001u) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageNoVirtualToArray(
        1, *this->pre_, deterministic, target);
  }

  // required .medical_imaging.MsgPoint2 cur = 2;
  if (cached_has_bits & 0x00000002u) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageNoVirtualToArray(
        2, *this->cur_, deterministic, target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:medical_imaging.MsgRotate)
  return target;
}

size_t MsgRotate::RequiredFieldsByteSizeFallback() const {
// @@protoc_insertion_point(required_fields_byte_size_fallback_start:medical_imaging.MsgRotate)
  size_t total_size = 0;

  if (has_pre()) {
    // required .medical_imaging.MsgPoint2 pre = 1;
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->pre_);
  }

  if (has_cur()) {
    // required .medical_imaging.MsgPoint2 cur = 2;
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->cur_);
  }

  return total_size;
}
size_t MsgRotate::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:medical_imaging.MsgRotate)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  if (((_has_bits_[0] & 0x00000003) ^ 0x00000003) == 0) {  // All required fields are present.
    // required .medical_imaging.MsgPoint2 pre = 1;
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->pre_);

    // required .medical_imaging.MsgPoint2 cur = 2;
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->cur_);

  } else {
    total_size += RequiredFieldsByteSizeFallback();
  }
  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void MsgRotate::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:medical_imaging.MsgRotate)
  GOOGLE_DCHECK_NE(&from, this);
  const MsgRotate* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const MsgRotate>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:medical_imaging.MsgRotate)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:medical_imaging.MsgRotate)
    MergeFrom(*source);
  }
}

void MsgRotate::MergeFrom(const MsgRotate& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:medical_imaging.MsgRotate)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 3u) {
    if (cached_has_bits & 0x00000001u) {
      mutable_pre()->::medical_imaging::MsgPoint2::MergeFrom(from.pre());
    }
    if (cached_has_bits & 0x00000002u) {
      mutable_cur()->::medical_imaging::MsgPoint2::MergeFrom(from.cur());
    }
  }
}

void MsgRotate::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:medical_imaging.MsgRotate)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void MsgRotate::CopyFrom(const MsgRotate& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:medical_imaging.MsgRotate)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MsgRotate::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000003) != 0x00000003) return false;
  if (has_pre()) {
    if (!this->pre_->IsInitialized()) return false;
  }
  if (has_cur()) {
    if (!this->cur_->IsInitialized()) return false;
  }
  return true;
}

void MsgRotate::Swap(MsgRotate* other) {
  if (other == this) return;
  InternalSwap(other);
}
void MsgRotate::InternalSwap(MsgRotate* other) {
  std::swap(pre_, other->pre_);
  std::swap(cur_, other->cur_);
  std::swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata MsgRotate::GetMetadata() const {
  protobuf_mi_5fmessage_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_mi_5fmessage_2eproto::file_level_metadata[kIndexInFileMessages];
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// MsgRotate

// required .medical_imaging.MsgPoint2 pre = 1;
bool MsgRotate::has_pre() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
void MsgRotate::set_has_pre() {
  _has_bits_[0] |= 0x00000001u;
}
void MsgRotate::clear_has_pre() {
  _has_bits_[0] &= ~0x00000001u;
}
void MsgRotate::clear_pre() {
  if (pre_ != NULL) pre_->::medical_imaging::MsgPoint2::Clear();
  clear_has_pre();
}
const ::medical_imaging::MsgPoint2& MsgRotate::pre() const {
  // @@protoc_insertion_point(field_get:medical_imaging.MsgRotate.pre)
  return pre_ != NULL ? *pre_
                         : *::medical_imaging::MsgPoint2::internal_default_instance();
}
::medical_imaging::MsgPoint2* MsgRotate::mutable_pre() {
  set_has_pre();
  if (pre_ == NULL) {
    pre_ = new ::medical_imaging::MsgPoint2;
  }
  // @@protoc_insertion_point(field_mutable:medical_imaging.MsgRotate.pre)
  return pre_;
}
::medical_imaging::MsgPoint2* MsgRotate::release_pre() {
  // @@protoc_insertion_point(field_release:medical_imaging.MsgRotate.pre)
  clear_has_pre();
  ::medical_imaging::MsgPoint2* temp = pre_;
  pre_ = NULL;
  return temp;
}
void MsgRotate::set_allocated_pre(::medical_imaging::MsgPoint2* pre) {
  delete pre_;
  pre_ = pre;
  if (pre) {
    set_has_pre();
  } else {
    clear_has_pre();
  }
  // @@protoc_insertion_point(field_set_allocated:medical_imaging.MsgRotate.pre)
}

// required .medical_imaging.MsgPoint2 cur = 2;
bool MsgRotate::has_cur() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
void MsgRotate::set_has_cur() {
  _has_bits_[0] |= 0x00000002u;
}
void MsgRotate::clear_has_cur() {
  _has_bits_[0] &= ~0x00000002u;
}
void MsgRotate::clear_cur() {
  if (cur_ != NULL) cur_->::medical_imaging::MsgPoint2::Clear();
  clear_has_cur();
}
const ::medical_imaging::MsgPoint2& MsgRotate::cur() const {
  // @@protoc_insertion_point(field_get:medical_imaging.MsgRotate.cur)
  return cur_ != NULL ? *cur_
                         : *::medical_imaging::MsgPoint2::internal_default_instance();
}
::medical_imaging::MsgPoint2* MsgRotate::mutable_cur() {
  set_has_cur();
  if (cur_ == NULL) {
    cur_ = new ::medical_imaging::MsgPoint2;
  }
  // @@protoc_insertion_point(field_mutable:medical_imaging.MsgRotate.cur)
  return cur_;
}
::medical_imaging::MsgPoint2* MsgRotate::release_cur() {
  // @@protoc_insertion_point(field_release:medical_imaging.MsgRotate.cur)
  clear_has_cur();
  ::medical_imaging::MsgPoint2* temp = cur_;
  cur_ = NULL;
  return temp;
}
void MsgRotate::set_allocated_cur(::medical_imaging::MsgPoint2* cur) {
  delete cur_;
  cur_ = cur;
  if (cur) {
    set_has_cur();
  } else {
    clear_has_cur();
  }
  // @@protoc_insertion_point(field_set_allocated:medical_imaging.MsgRotate.cur)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace medical_imaging

// @@protoc_insertion_point(global_scope)
