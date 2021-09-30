//
// Created by tesla on 2021/6/27.
//

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <grpc/grpc.h>
#include <grpc++/grpc++.h>
#include "image.pb.h"
#include "image.grpc.pb.h"
#include <sys/time.h>
#include <opencv2/opencv.hpp>

using std::map;
using std::vector;
using std::string;
using std::unique_ptr;
using std::to_string;
using std::cout;
using std::endl;
using cv::Mat;


struct Camera
{
    string grpc_ipport;
    int num;
};


class ImageFetcher
{
public:
    explicit ImageFetcher(map<int, Camera>& cameras): cameras(cameras) {}

    bool init()
    {
        auto rpcargs = grpc::ChannelArguments();
        rpcargs.SetMaxReceiveMessageSize(1<<30);
        rpcargs.SetInt(GRPC_ARG_MIN_RECONNECT_BACKOFF_MS, 5*1000);
        rpcargs.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 10*1000);
        rpcargs.SetInt(GRPC_ARG_INITIAL_RECONNECT_BACKOFF_MS, 1*1000);
        for (auto iter = cameras.begin(); iter != cameras.end(); ++iter)
        {
            cout << "Initializing the " << iter->first << " camera ..." << endl;
            char* grpc_ipport = const_cast<char*>(iter->second.grpc_ipport.c_str());
            auto channel = grpc::CreateCustomChannel(grpc_ipport, grpc::InsecureChannelCredentials(), rpcargs);
            stubs[iter->first] = SendImage::ImageServer::NewStub(channel);
            grpcidx[iter->first] = "0";
        }
        return true;
    }

    bool fetch(const int camid)
    {
        SendImage::GetImageRequest request;
        SendImage::GetImageReply reply;
        grpc::ClientContext context;
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
        context.set_deadline(deadline);
        cout << "Trying to fetch image of camera " << camid << endl;
        request.set_lastindex(grpcidx[camid]);
        request.set_cameraid(to_string(camid).c_str());
        grpc::Status status = stubs[camid]->GetImage(&context, request, &reply);
        cout << "Its status is " << status.ok() << " and its rows is " << reply.rows() << endl;
        if (!status.ok() || reply.rows() == 0) return false;
        cout << "Settinig grpc index, the vi index is " << reply.index() << endl;
        grpcidx[camid] = reply.index();
        cout << "Getting time of vi ..." << endl;
        string time_vi = timestamp(grpcidx[camid]);
        cout << "Getting the image id ..." << endl;
        string img_id = reply.refer();
        cout << "Getting the image data ..." << endl;
        Mat img = Mat(reply.rows(), reply.cols(), CV_8UC3, const_cast<char*>(reply.picture().c_str())).clone();
        cout << "Setting the image name ...." << endl;
        string svpath = "data/fetimg/" + to_string(camid) + "/" + to_string(cameras[camid].num) + ".jpg";
        cout << "Saving the image data, its path is " << svpath << endl;
        cv::imwrite(svpath, img);
        cout << "Fetched the " << cameras[camid].num << "th image of camera " << camid << endl;
        cameras[camid].num += 1;
        return true;
    }

    static string timestamp(const string& idx)
    {
        char stamp[100];
        struct tm p{};
        string usec;
        if (!idx.empty())
        {
            time_t t = atol(idx.c_str()) / 1000;
            p = *localtime(&t);
            usec = idx.substr(idx.size()-3);
        }
        else
        {
            struct timeval tv{};
            gettimeofday(&tv, nullptr);
            localtime_r(&tv.tv_sec, &p);
            usec = to_string(tv.tv_usec);
        }
        sprintf(stamp, "%02d-%02d-%02dT%02d:%02d:%02d.%s000", 1900+p.tm_year,
                1+p.tm_mon, p.tm_mday, p.tm_hour, p.tm_min, p.tm_sec, usec.c_str());
        return stamp;
    }


private:
    map<int, Camera> cameras;
    map<int, string> grpcidx;
    map<int, unique_ptr<SendImage::ImageServer::Stub>> stubs;
};


int main()
{
    vector<int> camids = {100, 101};
    map<int, Camera> cameras = {{100, {"172.18.192.14:50051", 0}},
                                {101, {"172.18.192.14:50051", 0}}};
    ImageFetcher fetcher(cameras);
    fetcher.init();
    for(int c = 0; c <= 16; ++c)
        for(int& camid: camids)
            fetcher.fetch(camid);
    return 0;
}
