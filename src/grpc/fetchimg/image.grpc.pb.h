// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: image.proto
#ifndef GRPC_image_2eproto__INCLUDED
#define GRPC_image_2eproto__INCLUDED

#include "image.pb.h"

#include <functional>
#include <grpc/impl/codegen/port_platform.h>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace SendImage {

// The greeting service definition.
class ImageServer final {
 public:
  static constexpr char const* service_full_name() {
    return "SendImage.ImageServer";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status GetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::SendImage::GetImageReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::SendImage::GetImageReply>> AsyncGetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::SendImage::GetImageReply>>(AsyncGetImageRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::SendImage::GetImageReply>> PrepareAsyncGetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::SendImage::GetImageReply>>(PrepareAsyncGetImageRaw(context, request, cq));
    }
    class experimental_async_interface {
     public:
      virtual ~experimental_async_interface() {}
      virtual void GetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest* request, ::SendImage::GetImageReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void GetImage(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::SendImage::GetImageReply* response, std::function<void(::grpc::Status)>) = 0;
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      virtual void GetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest* request, ::SendImage::GetImageReply* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      #else
      virtual void GetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest* request, ::SendImage::GetImageReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      #endif
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      virtual void GetImage(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::SendImage::GetImageReply* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      #else
      virtual void GetImage(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::SendImage::GetImageReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      #endif
    };
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    typedef class experimental_async_interface async_interface;
    #endif
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    async_interface* async() { return experimental_async(); }
    #endif
    virtual class experimental_async_interface* experimental_async() { return nullptr; }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::SendImage::GetImageReply>* AsyncGetImageRaw(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::SendImage::GetImageReply>* PrepareAsyncGetImageRaw(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status GetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::SendImage::GetImageReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::SendImage::GetImageReply>> AsyncGetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::SendImage::GetImageReply>>(AsyncGetImageRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::SendImage::GetImageReply>> PrepareAsyncGetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::SendImage::GetImageReply>>(PrepareAsyncGetImageRaw(context, request, cq));
    }
    class experimental_async final :
      public StubInterface::experimental_async_interface {
     public:
      void GetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest* request, ::SendImage::GetImageReply* response, std::function<void(::grpc::Status)>) override;
      void GetImage(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::SendImage::GetImageReply* response, std::function<void(::grpc::Status)>) override;
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      void GetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest* request, ::SendImage::GetImageReply* response, ::grpc::ClientUnaryReactor* reactor) override;
      #else
      void GetImage(::grpc::ClientContext* context, const ::SendImage::GetImageRequest* request, ::SendImage::GetImageReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      #endif
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      void GetImage(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::SendImage::GetImageReply* response, ::grpc::ClientUnaryReactor* reactor) override;
      #else
      void GetImage(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::SendImage::GetImageReply* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      #endif
     private:
      friend class Stub;
      explicit experimental_async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class experimental_async_interface* experimental_async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class experimental_async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::SendImage::GetImageReply>* AsyncGetImageRaw(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::SendImage::GetImageReply>* PrepareAsyncGetImageRaw(::grpc::ClientContext* context, const ::SendImage::GetImageRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_GetImage_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status GetImage(::grpc::ServerContext* context, const ::SendImage::GetImageRequest* request, ::SendImage::GetImageReply* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_GetImage : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_GetImage() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_GetImage() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetImage(::grpc::ServerContext* /*context*/, const ::SendImage::GetImageRequest* /*request*/, ::SendImage::GetImageReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetImage(::grpc::ServerContext* context, ::SendImage::GetImageRequest* request, ::grpc::ServerAsyncResponseWriter< ::SendImage::GetImageReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_GetImage<Service > AsyncService;
  template <class BaseClass>
  class ExperimentalWithCallbackMethod_GetImage : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    ExperimentalWithCallbackMethod_GetImage() {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::Service::
    #else
      ::grpc::Service::experimental().
    #endif
        MarkMethodCallback(0,
          new ::grpc_impl::internal::CallbackUnaryHandler< ::SendImage::GetImageRequest, ::SendImage::GetImageReply>(
            [this](
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
                   ::grpc::CallbackServerContext*
    #else
                   ::grpc::experimental::CallbackServerContext*
    #endif
                     context, const ::SendImage::GetImageRequest* request, ::SendImage::GetImageReply* response) { return this->GetImage(context, request, response); }));}
    void SetMessageAllocatorFor_GetImage(
        ::grpc::experimental::MessageAllocator< ::SendImage::GetImageRequest, ::SendImage::GetImageReply>* allocator) {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(0);
    #else
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::experimental().GetHandler(0);
    #endif
      static_cast<::grpc_impl::internal::CallbackUnaryHandler< ::SendImage::GetImageRequest, ::SendImage::GetImageReply>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~ExperimentalWithCallbackMethod_GetImage() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetImage(::grpc::ServerContext* /*context*/, const ::SendImage::GetImageRequest* /*request*/, ::SendImage::GetImageReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    virtual ::grpc::ServerUnaryReactor* GetImage(
      ::grpc::CallbackServerContext* /*context*/, const ::SendImage::GetImageRequest* /*request*/, ::SendImage::GetImageReply* /*response*/)
    #else
    virtual ::grpc::experimental::ServerUnaryReactor* GetImage(
      ::grpc::experimental::CallbackServerContext* /*context*/, const ::SendImage::GetImageRequest* /*request*/, ::SendImage::GetImageReply* /*response*/)
    #endif
      { return nullptr; }
  };
  #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
  typedef ExperimentalWithCallbackMethod_GetImage<Service > CallbackService;
  #endif

  typedef ExperimentalWithCallbackMethod_GetImage<Service > ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_GetImage : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_GetImage() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_GetImage() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetImage(::grpc::ServerContext* /*context*/, const ::SendImage::GetImageRequest* /*request*/, ::SendImage::GetImageReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_GetImage : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_GetImage() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_GetImage() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetImage(::grpc::ServerContext* /*context*/, const ::SendImage::GetImageRequest* /*request*/, ::SendImage::GetImageReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetImage(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class ExperimentalWithRawCallbackMethod_GetImage : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    ExperimentalWithRawCallbackMethod_GetImage() {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::Service::
    #else
      ::grpc::Service::experimental().
    #endif
        MarkMethodRawCallback(0,
          new ::grpc_impl::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
                   ::grpc::CallbackServerContext*
    #else
                   ::grpc::experimental::CallbackServerContext*
    #endif
                     context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->GetImage(context, request, response); }));
    }
    ~ExperimentalWithRawCallbackMethod_GetImage() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetImage(::grpc::ServerContext* /*context*/, const ::SendImage::GetImageRequest* /*request*/, ::SendImage::GetImageReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    virtual ::grpc::ServerUnaryReactor* GetImage(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)
    #else
    virtual ::grpc::experimental::ServerUnaryReactor* GetImage(
      ::grpc::experimental::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)
    #endif
      { return nullptr; }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_GetImage : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_GetImage() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< ::SendImage::GetImageRequest, ::SendImage::GetImageReply>(std::bind(&WithStreamedUnaryMethod_GetImage<BaseClass>::StreamedGetImage, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_GetImage() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status GetImage(::grpc::ServerContext* /*context*/, const ::SendImage::GetImageRequest* /*request*/, ::SendImage::GetImageReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedGetImage(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::SendImage::GetImageRequest,::SendImage::GetImageReply>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_GetImage<Service > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_GetImage<Service > StreamedService;
};

}  // namespace SendImage


#endif  // GRPC_image_2eproto__INCLUDED
