#include <map>
#include <string>
#include "utils/card.h"

static const std::string siftsmall_bpath = "/home/chunxy/datasets/siftsmall/siftsmall_base.fvecs";
static const std::string siftsmall_qpath = "/home/chunxy/datasets/siftsmall/siftsmall_query.fvecs";
static const std::string siftsmall_gpath = "/home/chunxy/datasets/siftsmall/siftsmall_groundtruth.ivecs";

static const std::string sift_bpath = "/home/chunxy/datasets/sift/sift_base.fvecs";
static const std::string sift_qpath = "/home/chunxy/datasets/sift/sift_query.fvecs";
static const std::string sift_gpath = "/home/chunxy/datasets/sift/sift_groundtruth.ivecs";

static const std::string gist_bpath = "/home/chunxy/datasets/gist/gist_base.fvecs";
static const std::string gist_qpath = "/home/chunxy/datasets/gist/gist_query.fvecs";
static const std::string gist_gpath = "/home/chunxy/datasets/gist/gist_groundtruth.ivecs";

static const std::string crawl_bpath = "/home/chunxy/datasets/crawl/crawl_base.fvecs";
static const std::string crawl_qpath = "/home/chunxy/datasets/crawl/crawl_query.fvecs";
static const std::string crawl_gpath = "/home/chunxy/datasets/crawl/crawl_groundtruth.ivecs";

static const std::string glove100_bpath = "/home/chunxy/datasets/glove100/glove100_base.fvecs";
static const std::string glove100_qpath = "/home/chunxy/datasets/glove100/glove100_query.fvecs";
static const std::string glove100_gpath = "/home/chunxy/datasets/glove100/glove100_groundtruth.ivecs";

static const std::string audio_bpath = "/home/chunxy/datasets/audio/audio_base.fvecs";
static const std::string audio_qpath = "/home/chunxy/datasets/audio/audio_query.fvecs";
static const std::string audio_gpath = "/home/chunxy/datasets/audio/audio_groundtruth.ivecs";

static const std::string video_bpath = "/home/chunxy/datasets/video/video_base.fvecs";
static const std::string video_qpath = "/home/chunxy/datasets/video/video_query.fvecs";
static const std::string video_gpath = "/home/chunxy/datasets/video/video_groundtruth.ivecs";

static const std::string audio_dedup_bpath = "/home/chunxy/datasets/audio-dedup/audio-dedup_base.fvecs";
static const std::string audio_dedup_qpath = "/home/chunxy/datasets/audio-dedup/audio-dedup_query.fvecs";
static const std::string audio_dedup_gpath = "/home/chunxy/datasets/audio-dedup/audio-dedup_groundtruth.ivecs";

static const std::string video_dedup_bpath = "/home/chunxy/datasets/video-dedup/video-dedup_base.fvecs";
static const std::string video_dedup_qpath = "/home/chunxy/datasets/video-dedup/video-dedup_query.fvecs";
static const std::string video_dedup_gpath = "/home/chunxy/datasets/video-dedup/video-dedup_groundtruth.ivecs";

static const std::string sift_dedup_bpath = "/home/chunxy/datasets/sift-dedup/sift-dedup_base.fvecs";
static const std::string sift_dedup_qpath = "/home/chunxy/datasets/sift-dedup/sift-dedup_query.fvecs";
static const std::string sift_dedup_gpath = "/home/chunxy/datasets/sift-dedup/sift-dedup_groundtruth.ivecs";

static const std::string gist_dedup_bpath = "/home/chunxy/datasets/gist-dedup/gist-dedup_base.fvecs";
static const std::string gist_dedup_qpath = "/home/chunxy/datasets/gist-dedup/gist-dedup_query.fvecs";
static const std::string gist_dedup_gpath = "/home/chunxy/datasets/gist-dedup/gist-dedup_groundtruth.ivecs";

static const std::string flickr_bpath = "/opt/nfs_dcc/chunxy/SVS/flickr/flickr_base.fvecs";
static const std::string flickr_qpath = "/opt/nfs_dcc/chunxy/SVS/flickr/flickr_query.fvecs";
static const std::string flickr_gpath = "/opt/nfs_dcc/chunxy/SVS/flickr/flickr_groundtruth.ivecs";

static const std::string deep10m_bpath = "/opt/nfs_dcc/chunxy/datasets/deep10m/deep10m_base.fvecs";
static const std::string deep10m_qpath = "/opt/nfs_dcc/chunxy/datasets/deep10m/deep10m_query.fvecs";
static const std::string deep10m_gpath = "/opt/nfs_dcc/chunxy/datasets/deep10m/deep10m_groundtruth.ivecs";

static const std::string word2vec_bpath = "/opt/nfs_dcc/chunxy/datasets/word2vec/word2vec_base.fvecs";
static const std::string word2vec_qpath = "/opt/nfs_dcc/chunxy/datasets/word2vec/word2vec_query.fvecs";
static const std::string word2vec_gpath = "/opt/nfs_dcc/chunxy/datasets/word2vec/word2vec_groundtruth.ivecs";

DataCard siftsmall_1_10_int32{
    "siftsmall",
    siftsmall_bpath,
    siftsmall_qpath,
    siftsmall_gpath,
    128,
    10000,
    100,
    100,
    1,
    10,
    "int32",
};

DataCard siftsmall_1_1000_int32{
    "siftsmall",
    siftsmall_bpath,
    siftsmall_qpath,
    siftsmall_gpath,
    128,
    10000,
    100,
    100,
    1,
    1000,
    "int32",
};

DataCard siftsmall_1_100_float32{
    "siftsmall",
    siftsmall_bpath,
    siftsmall_qpath,
    siftsmall_gpath,
    128,
    10000,
    100,
    100,
    1,
    100,
    "float32"
};

DataCard siftsmall_1_1000_float32{
    "siftsmall",
    siftsmall_bpath,
    siftsmall_qpath,
    siftsmall_gpath,
    128,
    10000,
    100,
    100,
    1,
    1000,
    "float32",
};

DataCard siftsmall_1_1000_top500_float32{
    "siftsmall",
    siftsmall_bpath,
    siftsmall_qpath,
    "/home/chunxy/repos/Compass/data/gt/siftsmall_1000_{0}_{1000}_500.hybrid.gt",
    128,
    10000,
    100,
    500,
    1,
    1000,
    "float32",
};

DataCard siftsmall_2_1000_float32{
    "siftsmall",
    siftsmall_bpath,
    siftsmall_qpath,
    siftsmall_gpath,
    128,
    10000,
    100,
    100,
    2,
    1000,
    "float32",
};

DataCard sift_1_2_int32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    2,
    "int32",
};

DataCard sift_1_5_int32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    5,
    "int32",
};

DataCard sift_1_10_int32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    10,
    "int32",
};

DataCard sift_1_20_int32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    20,
    "int32",
};

DataCard sift_1_50_int32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    50,
    "int32",
};

DataCard sift_1_100_int32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    100,
    "int32",
};

DataCard sift_1_10000_float32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    10000,
    "float32",
};

DataCard sift_2_10000_float32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    2,
    10000,
    "float32",
};

DataCard sift_3_10000_float32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    3,
    10000,
    "float32",
};

DataCard sift_4_10000_float32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    4,
    10000,
    "float32",
};

DataCard sift_5_10000_float32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    5,
    10000,
    "float32",
};

DataCard sift_6_10000_float32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    6,
    10000,
    "float32",
};

DataCard gist_1_2_int32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    2,
    "int32",
};

DataCard gist_1_5_int32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    5,
    "int32",
};

DataCard gist_1_10_int32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    10,
    "int32",
};

DataCard gist_1_20_int32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    20,
    "int32",
};

DataCard gist_1_50_int32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    50,
    "int32",
};

DataCard gist_1_100_int32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    100,
    "int32",
};

DataCard gist_1_10000_float32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    10000,
    "float32",
};

DataCard gist_2_10000_float32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    2,
    10000,
    "float32",
};

DataCard gist_3_10000_float32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    3,
    10000,
    "float32",
};

DataCard gist_4_10000_float32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    4,
    10000,
    "float32",
};

DataCard gist_5_10000_float32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    5,
    10000,
    "float32",
};

DataCard gist_6_10000_float32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    6,
    10000,
    "float32",
};

DataCard crawl_1_2_int32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    2,
    "int32",
};

DataCard crawl_1_5_int32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    5,
    "int32",
};

DataCard crawl_1_10_int32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    10,
    "int32",
};

DataCard crawl_1_20_int32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    20,
    "int32",
};

DataCard crawl_1_50_int32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    50,
    "int32",
};

DataCard crawl_1_100_int32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    100,
    "int32",
};

DataCard crawl_1_10000_float32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    10000,
    "float32",
};

DataCard crawl_2_10000_float32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    2,
    10000,
    "float32",
};

DataCard crawl_3_10000_float32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    3,
    10000,
    "float32",
};

DataCard crawl_4_10000_float32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    4,
    10000,
    "float32",
};

DataCard crawl_5_10000_float32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    5,
    10000,
    "float32",
};

DataCard crawl_6_10000_float32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    6,
    10000,
    "float32",
};

DataCard glove100_1_2_int32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    2,
    "int32",
};

DataCard glove100_1_5_int32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    5,
    "int32",
};

DataCard glove100_1_10_int32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    10,
    "int32",
};

DataCard glove100_1_20_int32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    20,
    "int32",
};

DataCard glove100_1_50_int32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    50,
    "int32",
};

DataCard glove100_1_100_int32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    100,
    "int32",
};

DataCard glove100_1_10000_float32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    10000,
    "float32",
};

DataCard glove100_2_10000_float32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    2,
    10000,
    "float32",
};

DataCard glove100_3_10000_float32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    3,
    10000,
    "float32",
};

DataCard glove100_4_10000_float32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    4,
    10000,
    "float32",
};

DataCard glove100_5_10000_float32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    5,
    10000,
    "float32",
};

DataCard glove100_6_10000_float32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    6,
    10000,
    "float32",
};

DataCard audio_1_10000_float32{
    "audio",
    audio_bpath,
    audio_qpath,
    audio_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    10000,
    "float32",
};

DataCard audio_2_10000_float32{
    "audio",
    audio_bpath,
    audio_qpath,
    audio_gpath,
    128,
    1'000'000,
    10'000,
    100,
    2,
    10000,
    "float32",
};

DataCard audio_3_10000_float32{
    "audio",
    audio_bpath,
    audio_qpath,
    audio_gpath,
    128,
    1'000'000,
    10'000,
    100,
    3,
    10000,
    "float32",
};

DataCard audio_4_10000_float32{
    "audio",
    audio_bpath,
    audio_qpath,
    audio_gpath,
    128,
    1'000'000,
    10'000,
    100,
    4,
    10000,
    "float32",
};

DataCard audio_5_10000_float32{
    "audio",
    audio_bpath,
    audio_qpath,
    audio_gpath,
    128,
    1'000'000,
    10'000,
    100,
    5,
    10000,
    "float32",
};

DataCard audio_6_10000_float32{
    "audio",
    audio_bpath,
    audio_qpath,
    audio_gpath,
    128,
    1'000'000,
    10'000,
    100,
    6,
    10000,
    "float32",
};

DataCard video_1_10000_float32{
    "video",
    video_bpath,
    video_qpath,
    video_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    1,
    10000,
    "float32",
};

DataCard video_2_10000_float32{
    "video",
    video_bpath,
    video_qpath,
    video_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    2,
    10000,
    "float32",
};

DataCard video_3_10000_float32{
    "video",
    video_bpath,
    video_qpath,
    video_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    3,
    10000,
    "float32",
};

DataCard video_4_10000_float32{
    "video",
    video_bpath,
    video_qpath,
    video_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    4,
    10000,
    "float32",
};

DataCard video_5_10000_float32{
    "video",
    video_bpath,
    video_qpath,
    video_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    5,
    10000,
    "float32",
};

DataCard video_6_10000_float32{
    "video",
    video_bpath,
    video_qpath,
    video_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    6,
    10000,
    "float32",
};

DataCard sift_dedup_1_2_int32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    2,
    "int32",
};

DataCard sift_dedup_1_5_int32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    5,
    "int32",
};

DataCard sift_dedup_1_10_int32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    10,
    "int32",
};

DataCard sift_dedup_1_20_int32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    20,
    "int32",
};

DataCard sift_dedup_1_50_int32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    50,
    "int32",
};

DataCard sift_dedup_1_100_int32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    100,
    "int32",
};

DataCard audio_dedup_1_10000_float32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    10000,
    "float32",
};

DataCard audio_dedup_2_10000_float32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    2,
    10000,
    "float32",
};

DataCard audio_dedup_3_10000_float32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    3,
    10000,
    "float32",
};

DataCard audio_dedup_4_10000_float32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    4,
    10000,
    "float32",
};

DataCard audio_dedup_5_10000_float32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    5,
    10000,
    "float32",
};

DataCard audio_dedup_6_10000_float32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    6,
    10000,
    "float32",
};

DataCard audio_dedup_1_2_int32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    2,
    "int32",
};

DataCard audio_dedup_1_5_int32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    5,
    "int32",
};

DataCard audio_dedup_1_10_int32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    10,
    "int32",
};

DataCard audio_dedup_1_20_int32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    20,
    "int32",
};

DataCard audio_dedup_1_50_int32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    50,
    "int32",
};

DataCard audio_dedup_1_100_int32{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    100,
    "int32",
};

DataCard video_dedup_1_2_int32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    2,
    "int32",
};

DataCard video_dedup_1_5_int32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    5,
    "int32",
};

DataCard video_dedup_1_10_int32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    10,
    "int32",
};

DataCard video_dedup_1_20_int32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    20,
    "int32",
};

DataCard video_dedup_1_50_int32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    50,
    "int32",
};

DataCard video_dedup_1_100_int32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    100,
    "int32",
};

DataCard video_dedup_1_10000_float32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    1,
    10000,
    "float32",
};

DataCard video_dedup_2_10000_float32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    2,
    10000,
    "float32",
};

DataCard video_dedup_3_10000_float32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    3,
    10000,
    "float32",
};

DataCard video_dedup_4_10000_float32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    4,
    10000,
    "float32",
};

DataCard video_dedup_5_10000_float32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    5,
    10000,
    "float32",
};

DataCard video_dedup_6_10000_float32{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    1024,
    1'000'000,
    10'000,
    100,
    6,
    10000,
    "float32",
};

DataCard sift_dedup_1_10000_float32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    10000,
    "float32",
};

DataCard sift_dedup_2_10000_float32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    2,
    10000,
    "float32",
};

DataCard sift_dedup_3_10000_float32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    3,
    10000,
    "float32",
};

DataCard sift_dedup_4_10000_float32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    4,
    10000,
    "float32",
};

DataCard sift_dedup_5_10000_float32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    5,
    10000,
    "float32",
};

DataCard sift_dedup_6_10000_float32{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10'000,
    100,
    6,
    10000,
    "float32",
};

DataCard gist_dedup_1_2_int32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    128,
    1'000'000 - 17306,
    10'000,
    100,
    1,
    2,
    "int32",
};

DataCard gist_dedup_1_5_int32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    128,
    1'000'000 - 17306,
    10'000,
    100,
    1,
    5,
    "int32",
};

DataCard gist_dedup_1_10_int32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    128,
    1'000'000 - 17306,
    10'000,
    100,
    1,
    10,
    "int32",
};

DataCard gist_dedup_1_20_int32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    128,
    1'000'000 - 17306,
    10'000,
    100,
    1,
    20,
    "int32",
};

DataCard gist_dedup_1_50_int32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    128,
    1'000'000 - 17306,
    10'000,
    100,
    1,
    50,
    "int32",
};

DataCard gist_dedup_1_100_int32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    128,
    1'000'000 - 17306,
    10'000,
    100,
    1,
    100,
    "int32",
};

DataCard gist_dedup_1_10000_float32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    960,
    1'000'000 - 17306,
    1'000,
    100,
    1,
    10000,
    "float32",
};

DataCard gist_dedup_2_10000_float32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    960,
    1'000'000 - 17306,
    1'000,
    100,
    2,
    10000,
    "float32",
};

DataCard gist_dedup_3_10000_float32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    960,
    1'000'000 - 17306,
    1'000,
    100,
    3,
    10000,
    "float32",
};

DataCard gist_dedup_4_10000_float32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    960,
    1'000'000 - 17306,
    1'000,
    100,
    4,
    10000,
    "float32",
};

DataCard gist_dedup_5_10000_float32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    960,
    1'000'000 - 17306,
    1'000,
    100,
    5,
    10000,
    "float32",
};

DataCard gist_dedup_6_10000_float32{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    960,
    1'000'000 - 17306,
    1'000,
    100,
    6,
    10000,
    "float32",
};

DataCard flickr_1_10000_float32{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    flickr_gpath,
    512,
    4203901,
    29999,
    100,
    1,
    10000,
    "float32",
};

DataCard flickr_2_10000_float32{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    flickr_gpath,
    512,
    4203901,
    29999,
    100,
    2,
    10000,
    "float32",
};

DataCard flickr_3_10000_float32{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    flickr_gpath,
    512,
    4203901,
    29999,
    100,
    3,
    10000,
    "float32",
};

DataCard flickr_4_10000_float32{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    flickr_gpath,
    512,
    4203901,
    29999,
    100,
    4,
    10000,
    "float32",
};

DataCard deep10m_1_10000_float32{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    deep10m_gpath,
    96,
    10000000,
    10000,
    100,
    1,
    10000,
    "float32",
};

DataCard deep10m_2_10000_float32{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    deep10m_gpath,
    96,
    10000000,
    10000,
    100,
    2,
    10000,
    "float32",
};

DataCard deep10m_3_10000_float32{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    deep10m_gpath,
    96,
    10000000,
    10000,
    100,
    3,
    10000,
    "float32",
};

DataCard deep10m_4_10000_float32{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    deep10m_gpath,
    96,
    10000000,
    10000,
    100,
    4,
    10000,
    "float32",
};

DataCard word2vec_1_10000_float32{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    word2vec_gpath,
    300,
    1000000,
    1000,
    100,
    1,
    10000,
    "float32",
};

DataCard word2vec_2_10000_float32{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    word2vec_gpath,
    300,
    1000000,
    1000,
    100,
    2,
    10000,
    "float32",
};

DataCard word2vec_3_10000_float32{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    word2vec_gpath,
    300,
    1000000,
    1000,
    100,
    3,
    10000,
    "float32",
};

DataCard word2vec_4_10000_float32{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    word2vec_gpath,
    300,
    1000000,
    1000,
    100,
    4,
    10000,
    "float32",
};

DataCard sift_dedup_1_30_float32_skewed{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/sift-dedup_1_30.skewed.hybrid.gt",
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/sift-dedup_1_30.skewed.value.bin",
    "skewed"
};

DataCard sift_dedup_2_20_float32_correlated{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/sift-dedup_2_20.correlated.hybrid.gt",
    128,
    1'000'000 - 14538,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/sift-dedup_2_20.correlated.value.bin",
    "correlated"
};

DataCard sift_dedup_2_20_float32_anticorrelated{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/sift-dedup_2_20.anticorrelated.hybrid.gt",
    128,
    1'000'000 - 14538,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/sift-dedup_2_20.anticorrelated.value.bin",
    "anticorrelated"
};

DataCard sift_dedup_1_30_float32_onesided{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/sift-dedup_1_30.onesided.hybrid.gt",
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/sift-dedup_1_30.skewed.value.bin",
    "onesided"
};

DataCard sift_dedup_1_30_float32_point{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/sift-dedup_1_30.point.hybrid.gt",
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/sift-dedup_1_30.skewed.value.bin",
    "point"
};

DataCard sift_dedup_1_30_float32_negation{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/sift-dedup_1_30.negation.hybrid.gt",
    128,
    1'000'000 - 14538,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/sift-dedup_1_30.skewed.value.bin",
    "negation"
};

DataCard audio_dedup_1_30_float32_skewed{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/audio-dedup_1_30.skewed.hybrid.gt",
    128,
    1'000'000,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/audio-dedup_1_30.skewed.value.bin",
    "skewed"
};

DataCard audio_dedup_2_20_float32_correlated{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/audio-dedup_2_20.correlated.hybrid.gt",
    128,
    1'000'000,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/audio-dedup_2_20.correlated.value.bin",
    "correlated"
};

DataCard audio_dedup_2_20_float32_anticorrelated{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/audio-dedup_2_20.anticorrelated.hybrid.gt",
    128,
    1'000'000,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/audio-dedup_2_20.anticorrelated.value.bin",
    "anticorrelated"
};

DataCard audio_dedup_1_30_float32_onesided{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/audio-dedup_1_30.onesided.hybrid.gt",
    128,
    1'000'000,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/audio-dedup_1_30.skewed.value.bin",
    "onesided"
};

DataCard audio_dedup_1_30_float32_point{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/audio-dedup_1_30.point.hybrid.gt",
    128,
    1'000'000,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/audio-dedup_1_30.skewed.value.bin",
    "point"
};

DataCard audio_dedup_1_30_float32_negation{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/audio-dedup_1_30.negation.hybrid.gt",
    128,
    1'000'000,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/audio-dedup_1_30.skewed.value.bin",
    "negation"
};

DataCard video_dedup_1_30_float32_skewed{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/video-dedup_1_30.skewed.hybrid.gt",
    1024,
    1'000'000,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/video-dedup_1_30.skewed.value.bin",
    "skewed"
};

DataCard video_dedup_2_20_float32_correlated{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/video-dedup_2_20.correlated.hybrid.gt",
    1024,
    1'000'000,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/video-dedup_2_20.correlated.value.bin",
    "correlated"
};

DataCard video_dedup_2_20_float32_anticorrelated{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/video-dedup_2_20.anticorrelated.hybrid.gt",
    1024,
    1'000'000,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/video-dedup_2_20.anticorrelated.value.bin",
    "anticorrelated"
};

DataCard video_dedup_1_30_float32_onesided{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/video-dedup_1_30.onesided.hybrid.gt",
    1024,
    1'000'000,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/video-dedup_1_30.skewed.value.bin",
    "onesided"
};

DataCard video_dedup_1_30_float32_point{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/video-dedup_1_30.point.hybrid.gt",
    1024,
    1'000'000,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/video-dedup_1_30.skewed.value.bin",
    "point"
};

DataCard video_dedup_1_30_float32_negation{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/video-dedup_1_30.negation.hybrid.gt",
    1024,
    1'000'000,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/video-dedup_1_30.skewed.value.bin",
    "negation"
};

DataCard gist_dedup_1_30_float32_skewed{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/gist-dedup_1_30.skewed.hybrid.gt",
    960,
    1'000'000 - 17306,
    1'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/gist-dedup_1_30.skewed.value.bin",
    "skewed"
};

DataCard gist_dedup_2_20_float32_correlated{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/gist-dedup_2_20.correlated.hybrid.gt",
    960,
    1'000'000 - 17306,
    1'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/gist-dedup_2_20.correlated.value.bin",
    "correlated"
};

DataCard gist_dedup_2_20_float32_anticorrelated{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/gist-dedup_2_20.anticorrelated.hybrid.gt",
    960,
    1'000'000 - 17306,
    1'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/gist-dedup_2_20.anticorrelated.value.bin",
    "anticorrelated"
};

DataCard gist_dedup_1_30_float32_onesided{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/gist-dedup_1_30.onesided.hybrid.gt",
    960,
    1'000'000 - 17306,
    1'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/gist-dedup_1_30.skewed.value.bin",
    "onesided"
};

DataCard gist_dedup_1_30_float32_point{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/gist-dedup_1_30.point.hybrid.gt",
    960,
    1'000'000 - 17306,
    1'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/gist-dedup_1_30.skewed.value.bin",
    "point"
};

DataCard gist_dedup_1_30_float32_negation{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/gist-dedup_1_30.negation.hybrid.gt",
    960,
    1'000'000 - 17306,
    1'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/gist-dedup_1_30.skewed.value.bin",
    "negation"
};

DataCard crawl_1_30_float32_skewed{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/crawl_1_30.skewed.hybrid.gt",
    300,
    1'989'995,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/crawl_1_30.skewed.value.bin",
    "skewed"
};

DataCard crawl_2_20_float32_correlated{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/crawl_2_20.correlated.hybrid.gt",
    300,
    1'989'995,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/crawl_2_20.correlated.value.bin",
    "correlated"
};

DataCard crawl_2_20_float32_anticorrelated{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/crawl_2_20.anticorrelated.hybrid.gt",
    300,
    1'989'995,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/crawl_2_20.anticorrelated.value.bin",
    "anticorrelated"
};

DataCard crawl_1_30_float32_onesided{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/crawl_1_30.onesided.hybrid.gt",
    300,
    1'989'995,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/crawl_1_30.skewed.value.bin",
    "onesided"
};

DataCard crawl_1_30_float32_point{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/crawl_1_30.point.hybrid.gt",
    300,
    1'989'995,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/crawl_1_30.skewed.value.bin",
    "point"
};

DataCard crawl_1_30_float32_negation{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/crawl_1_30.negation.hybrid.gt",
    300,
    1'989'995,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/crawl_1_30.skewed.value.bin",
    "negation"
};

DataCard glove100_1_30_float32_skewed{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/glove100_1_30.skewed.hybrid.gt",
    100,
    1'183'514,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/glove100_1_30.skewed.value.bin",
    "skewed"
};

DataCard glove100_2_20_float32_correlated{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/glove100_2_20.correlated.hybrid.gt",
    100,
    1'183'514,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/glove100_2_20.correlated.value.bin",
    "correlated"
};

DataCard glove100_2_20_float32_anticorrelated{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/glove100_2_20.anticorrelated.hybrid.gt",
    100,
    1'183'514,
    10'000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/glove100_2_20.anticorrelated.value.bin",
    "anticorrelated"
};

DataCard glove100_1_30_float32_onesided{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/glove100_1_30.onesided.hybrid.gt",
    100,
    1'183'514,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/glove100_1_30.skewed.value.bin",
    "onesided"
};

DataCard glove100_1_30_float32_point{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/glove100_1_30.point.hybrid.gt",
    100,
    1'183'514,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/glove100_1_30.skewed.value.bin",
    "point"
};

DataCard glove100_1_30_float32_negation{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/glove100_1_30.negation.hybrid.gt",
    100,
    1'183'514,
    10'000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/glove100_1_30.skewed.value.bin",
    "negation"
};

DataCard flickr_1_30_float32_skewed{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/flickr_1_30.skewed.hybrid.gt",
    512,
    4203901,
    29999,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/flickr_1_30.skewed.value.bin",
    "skewed"
};

DataCard flickr_2_20_float32_correlated{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/flickr_2_20.correlated.hybrid.gt",
    512,
    4203901,
    29999,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/flickr_2_20.correlated.value.bin",
    "correlated"
};


DataCard flickr_2_20_float32_anticorrelated{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/flickr_2_20.anticorrelated.hybrid.gt",
    512,
    4203901,
    29999,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/flickr_2_20.anticorrelated.value.bin",
    "anticorrelated"
};

DataCard flickr_1_30_float32_onesided{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/flickr_1_30.onesided.hybrid.gt",
    512,
    4203901,
    29999,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/flickr_1_30.skewed.value.bin",
    "onesided"
};

DataCard flickr_1_30_float32_point{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/flickr_1_30.point.hybrid.gt",
    512,
    4203901,
    29999,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/flickr_1_30.skewed.value.bin",
    "point"
};

DataCard flickr_1_30_float32_negation{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/flickr_1_30.negation.hybrid.gt",
    512,
    4203901,
    29999,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/flickr_1_30.skewed.value.bin",
    "negation"
};

DataCard deep10m_1_30_float32_skewed{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/deep10m_1_30.skewed.hybrid.gt",
    96,
    10000000,
    10000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/deep10m_1_30.skewed.value.bin",
    "skewed"
};

DataCard deep10m_2_20_float32_correlated{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/deep10m_2_20.correlated.hybrid.gt",
    96,
    10000000,
    10000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/deep10m_2_20.correlated.value.bin",
    "correlated"
};

DataCard deep10m_2_20_float32_anticorrelated{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/deep10m_2_20.anticorrelated.hybrid.gt",
    96,
    10000000,
    10000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/deep10m_2_20.anticorrelated.value.bin",
    "anticorrelated"
};

DataCard deep10m_1_30_float32_onesided{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/deep10m_1_30.onesided.hybrid.gt",
    96,
    10000000,
    10000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/deep10m_1_30.skewed.value.bin",
    "onesided"
};

DataCard deep10m_1_60_float32_onesided{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/deep10m_1_60.onesided.hybrid.gt",
    96,
    10000000,
    10000,
    100,
    1,
    60,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/deep10m_1_30.skewed.value.bin",
    "onesided"
};

DataCard deep10m_1_30_float32_point{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/deep10m_1_30.point.hybrid.gt",
    96,
    10000000,
    10000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/deep10m_1_30.skewed.value.bin",
    "point"
};

DataCard deep10m_1_30_float32_negation{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/deep10m_1_30.negation.hybrid.gt",
    96,
    10000000,
    10000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/deep10m_1_30.skewed.value.bin",
    "negation"
};

DataCard word2vec_1_30_float32_skewed{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/word2vec_1_30.skewed.hybrid.gt",
    300,
    1000000,
    1000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/word2vec_1_30.skewed.value.bin",
    "skewed"
};

DataCard word2vec_2_20_float32_correlated{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/word2vec_2_20.correlated.hybrid.gt",
    300,
    1000000,
    1000,
    100,
    2,
    20,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/word2vec_2_20.correlated.value.bin",
    "correlated"
};

DataCard word2vec_1_30_float32_onesided{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/word2vec_1_30.onesided.hybrid.gt",
    300,
    1000000,
    1000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/word2vec_1_30.skewed.value.bin",
    "onesided"
};

DataCard word2vec_1_30_float32_point{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/word2vec_1_30.point.hybrid.gt",
    300,
    1000000,
    1000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/word2vec_1_30.skewed.value.bin",
    "point"
};

DataCard word2vec_1_30_float32_negation{
    "word2vec",
    word2vec_bpath,
    word2vec_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/word2vec_1_30.negation.hybrid.gt",
    300,
    1000000,
    1000,
    100,
    1,
    30,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/word2vec_1_30.skewed.value.bin",
    "negation"
};

DataCard flickr_2_180_float32_real{
    "flickr",
    flickr_bpath,
    flickr_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/flickr_2_180.real.hybrid.gt",
    512,
    4203901,
    29999,
    100,
    2,
    180,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/flickr_2_180.real.value.bin",
    "real"
};

DataCard video_dedup_2_10000_float32_real{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    // reuse this field in revision
    "/home/chunxy/repos/Compass/data/gt/video-dedup_2_10000.real.hybrid.gt",
    1024,
    1'000'000,
    10'000,
    100,
    2,
    10000,
    "float32",
    // add this field in revision
    "/home/chunxy/repos/Compass/data/attr/video-dedup_2_10000.real.value.bin",
    "real"
};

std::map<std::string, DataCard> name_to_card{
    {"siftsmall_1_10_int32", siftsmall_1_10_int32},
    {"siftsmall_1_1000_int32", siftsmall_1_1000_int32},
    {"siftsmall_1_100_float32", siftsmall_1_100_float32},
    {"siftsmall_1_1000_float32", siftsmall_1_1000_float32},
    {"siftsmall_1_1000_top500_float32", siftsmall_1_1000_top500_float32},
    {"siftsmall_2_1000_float32", siftsmall_2_1000_float32},
    {"sift_1_2_int32", sift_1_2_int32},
    {"sift_1_5_int32", sift_1_5_int32},
    {"sift_1_10_int32", sift_1_10_int32},
    {"sift_1_20_int32", sift_1_20_int32},
    {"sift_1_50_int32", sift_1_50_int32},
    {"sift_1_100_int32", sift_1_100_int32},
    {"sift_1_10000_float32", sift_1_10000_float32},
    {"sift_2_10000_float32", sift_2_10000_float32},
    {"sift_3_10000_float32", sift_3_10000_float32},
    {"sift_4_10000_float32", sift_4_10000_float32},
    {"sift_5_10000_float32", sift_5_10000_float32},
    {"sift_6_10000_float32", sift_6_10000_float32},
    {"gist_1_2_int32", gist_1_2_int32},
    {"gist_1_5_int32", gist_1_5_int32},
    {"gist_1_10_int32", gist_1_10_int32},
    {"gist_1_20_int32", gist_1_20_int32},
    {"gist_1_50_int32", gist_1_50_int32},
    {"gist_1_100_int32", gist_1_100_int32},
    {"gist_1_10000_float32", gist_1_10000_float32},
    {"gist_2_10000_float32", gist_2_10000_float32},
    {"gist_3_10000_float32", gist_3_10000_float32},
    {"gist_4_10000_float32", gist_4_10000_float32},
    {"gist_5_10000_float32", gist_5_10000_float32},
    {"gist_6_10000_float32", gist_6_10000_float32},
    {"crawl_1_2_int32", crawl_1_2_int32},
    {"crawl_1_5_int32", crawl_1_5_int32},
    {"crawl_1_10_int32", crawl_1_10_int32},
    {"crawl_1_20_int32", crawl_1_20_int32},
    {"crawl_1_50_int32", crawl_1_50_int32},
    {"crawl_1_100_int32", crawl_1_100_int32},
    {"crawl_1_10000_float32", crawl_1_10000_float32},
    {"crawl_2_10000_float32", crawl_2_10000_float32},
    {"crawl_3_10000_float32", crawl_3_10000_float32},
    {"crawl_4_10000_float32", crawl_4_10000_float32},
    {"crawl_5_10000_float32", crawl_5_10000_float32},
    {"crawl_6_10000_float32", crawl_6_10000_float32},
    {"glove100_1_2_int32", glove100_1_2_int32},
    {"glove100_1_5_int32", glove100_1_5_int32},
    {"glove100_1_10_int32", glove100_1_10_int32},
    {"glove100_1_20_int32", glove100_1_20_int32},
    {"glove100_1_50_int32", glove100_1_50_int32},
    {"glove100_1_100_int32", glove100_1_100_int32},
    {"glove100_1_10000_float32", glove100_1_10000_float32},
    {"glove100_2_10000_float32", glove100_2_10000_float32},
    {"glove100_3_10000_float32", glove100_3_10000_float32},
    {"glove100_4_10000_float32", glove100_4_10000_float32},
    {"glove100_5_10000_float32", glove100_5_10000_float32},
    {"glove100_6_10000_float32", glove100_6_10000_float32},
    {"audio_1_10000_float32", audio_1_10000_float32},
    {"audio_2_10000_float32", audio_2_10000_float32},
    {"audio_3_10000_float32", audio_3_10000_float32},
    {"audio_4_10000_float32", audio_4_10000_float32},
    {"audio_5_10000_float32", audio_5_10000_float32},
    {"audio_6_10000_float32", audio_6_10000_float32},
    {"video_1_10000_float32", video_1_10000_float32},
    {"video_2_10000_float32", video_2_10000_float32},
    {"video_3_10000_float32", video_3_10000_float32},
    {"video_4_10000_float32", video_4_10000_float32},
    {"video_5_10000_float32", video_5_10000_float32},
    {"video_6_10000_float32", video_6_10000_float32},
    {"audio-dedup_1_2_int32", audio_dedup_1_2_int32},
    {"audio-dedup_1_5_int32", audio_dedup_1_5_int32},
    {"audio-dedup_1_10_int32", audio_dedup_1_10_int32},
    {"audio-dedup_1_20_int32", audio_dedup_1_20_int32},
    {"audio-dedup_1_50_int32", audio_dedup_1_50_int32},
    {"audio-dedup_1_100_int32", audio_dedup_1_100_int32},
    {"audio-dedup_1_10000_float32", audio_dedup_1_10000_float32},
    {"audio-dedup_2_10000_float32", audio_dedup_2_10000_float32},
    {"audio-dedup_3_10000_float32", audio_dedup_3_10000_float32},
    {"audio-dedup_4_10000_float32", audio_dedup_4_10000_float32},
    {"audio-dedup_5_10000_float32", audio_dedup_5_10000_float32},
    {"audio-dedup_6_10000_float32", audio_dedup_6_10000_float32},
    {"video-dedup_1_2_int32", video_dedup_1_2_int32},
    {"video-dedup_1_5_int32", video_dedup_1_5_int32},
    {"video-dedup_1_10_int32", video_dedup_1_10_int32},
    {"video-dedup_1_20_int32", video_dedup_1_20_int32},
    {"video-dedup_1_50_int32", video_dedup_1_50_int32},
    {"video-dedup_1_100_int32", video_dedup_1_100_int32},
    {"video-dedup_1_10000_float32", video_dedup_1_10000_float32},
    {"video-dedup_2_10000_float32", video_dedup_2_10000_float32},
    {"video-dedup_3_10000_float32", video_dedup_3_10000_float32},
    {"video-dedup_4_10000_float32", video_dedup_4_10000_float32},
    {"video-dedup_5_10000_float32", video_dedup_5_10000_float32},
    {"video-dedup_6_10000_float32", video_dedup_6_10000_float32},
    {"sift-dedup_1_2_int32", sift_dedup_1_2_int32},
    {"sift-dedup_1_5_int32", sift_dedup_1_5_int32},
    {"sift-dedup_1_10_int32", sift_dedup_1_10_int32},
    {"sift-dedup_1_20_int32", sift_dedup_1_20_int32},
    {"sift-dedup_1_50_int32", sift_dedup_1_50_int32},
    {"sift-dedup_1_100_int32", sift_dedup_1_100_int32},
    {"sift-dedup_1_10000_float32", sift_dedup_1_10000_float32},
    {"sift-dedup_2_10000_float32", sift_dedup_2_10000_float32},
    {"sift-dedup_3_10000_float32", sift_dedup_3_10000_float32},
    {"sift-dedup_4_10000_float32", sift_dedup_4_10000_float32},
    {"sift-dedup_5_10000_float32", sift_dedup_5_10000_float32},
    {"sift-dedup_6_10000_float32", sift_dedup_6_10000_float32},
    {"gist-dedup_1_2_int32", gist_dedup_1_2_int32},
    {"gist-dedup_1_5_int32", gist_dedup_1_5_int32},
    {"gist-dedup_1_10_int32", gist_dedup_1_10_int32},
    {"gist-dedup_1_20_int32", gist_dedup_1_20_int32},
    {"gist-dedup_1_50_int32", gist_dedup_1_50_int32},
    {"gist-dedup_1_100_int32", gist_dedup_1_100_int32},
    {"gist-dedup_1_10000_float32", gist_dedup_1_10000_float32},
    {"gist-dedup_2_10000_float32", gist_dedup_2_10000_float32},
    {"gist-dedup_3_10000_float32", gist_dedup_3_10000_float32},
    {"gist-dedup_4_10000_float32", gist_dedup_4_10000_float32},
    {"gist-dedup_5_10000_float32", gist_dedup_5_10000_float32},
    {"gist-dedup_6_10000_float32", gist_dedup_6_10000_float32},
    {"flickr_1_10000_float32", flickr_1_10000_float32},
    {"flickr_2_10000_float32", flickr_2_10000_float32},
    {"flickr_3_10000_float32", flickr_3_10000_float32},
    {"flickr_4_10000_float32", flickr_4_10000_float32},
    {"deep10m_1_10000_float32", deep10m_1_10000_float32},
    {"deep10m_2_10000_float32", deep10m_2_10000_float32},
    {"deep10m_3_10000_float32", deep10m_3_10000_float32},
    {"deep10m_4_10000_float32", deep10m_4_10000_float32},
    {"word2vec_1_10000_float32", word2vec_1_10000_float32},
    {"word2vec_2_10000_float32", word2vec_2_10000_float32},
    {"word2vec_3_10000_float32", word2vec_3_10000_float32},
    {"word2vec_4_10000_float32", word2vec_4_10000_float32},
    {"sift-dedup_1_30_float32_skewed", sift_dedup_1_30_float32_skewed},
    {"sift-dedup_2_20_float32_correlated", sift_dedup_2_20_float32_correlated},
    {"sift-dedup_2_20_float32_anticorrelated", sift_dedup_2_20_float32_anticorrelated},
    {"sift-dedup_1_30_float32_onesided", sift_dedup_1_30_float32_onesided},
    {"sift-dedup_1_30_float32_point", sift_dedup_1_30_float32_point},
    {"sift-dedup_1_30_float32_negation", sift_dedup_1_30_float32_negation},
    {"video-dedup_1_30_float32_skewed", video_dedup_1_30_float32_skewed},
    {"video-dedup_2_20_float32_correlated", video_dedup_2_20_float32_correlated},
    {"video-dedup_2_20_float32_anticorrelated", video_dedup_2_20_float32_anticorrelated},
    {"video-dedup_1_30_float32_onesided", video_dedup_1_30_float32_onesided},
    {"video-dedup_1_30_float32_point", video_dedup_1_30_float32_point},
    {"video-dedup_1_30_float32_negation", video_dedup_1_30_float32_negation},
    {"audio-dedup_1_30_float32_skewed", audio_dedup_1_30_float32_skewed},
    {"audio-dedup_2_20_float32_correlated", audio_dedup_2_20_float32_correlated},
    {"audio-dedup_2_20_float32_anticorrelated", audio_dedup_2_20_float32_anticorrelated},
    {"audio-dedup_1_30_float32_onesided", audio_dedup_1_30_float32_onesided},
    {"audio-dedup_1_30_float32_point", audio_dedup_1_30_float32_point},
    {"audio-dedup_1_30_float32_negation", audio_dedup_1_30_float32_negation},
    {"glove100_1_30_float32_skewed", glove100_1_30_float32_skewed},
    {"glove100_2_20_float32_correlated", glove100_2_20_float32_correlated},
    {"glove100_2_20_float32_anticorrelated", glove100_2_20_float32_anticorrelated},
    {"glove100_1_30_float32_onesided", glove100_1_30_float32_onesided},
    {"glove100_1_30_float32_point", glove100_1_30_float32_point},
    {"glove100_1_30_float32_negation", glove100_1_30_float32_negation},
    {"crawl_1_30_float32_skewed", crawl_1_30_float32_skewed},
    {"crawl_2_20_float32_correlated", crawl_2_20_float32_correlated},
    {"crawl_2_20_float32_anticorrelated", crawl_2_20_float32_anticorrelated},
    {"crawl_1_30_float32_onesided", crawl_1_30_float32_onesided},
    {"crawl_1_30_float32_point", crawl_1_30_float32_point},
    {"crawl_1_30_float32_negation", crawl_1_30_float32_negation},
    {"gist-dedup_1_30_float32_skewed", gist_dedup_1_30_float32_skewed},
    {"gist-dedup_2_20_float32_correlated", gist_dedup_2_20_float32_correlated},
    {"gist-dedup_2_20_float32_anticorrelated", gist_dedup_2_20_float32_anticorrelated},
    {"gist-dedup_1_30_float32_onesided", gist_dedup_1_30_float32_onesided},
    {"gist-dedup_1_30_float32_point", gist_dedup_1_30_float32_point},
    {"gist-dedup_1_30_float32_negation", gist_dedup_1_30_float32_negation},
    {"flickr_1_30_float32_skewed", flickr_1_30_float32_skewed},
    {"flickr_2_20_float32_correlated", flickr_2_20_float32_correlated},
    {"flickr_2_20_float32_anticorrelated", flickr_2_20_float32_anticorrelated},
    {"flickr_1_30_float32_onesided", flickr_1_30_float32_onesided},
    {"flickr_1_30_float32_point", flickr_1_30_float32_point},
    {"flickr_1_30_float32_negation", flickr_1_30_float32_negation},
    {"deep10m_1_30_float32_skewed", deep10m_1_30_float32_skewed},
    {"deep10m_2_20_float32_correlated", deep10m_2_20_float32_correlated},
    {"deep10m_2_20_float32_anticorrelated", deep10m_2_20_float32_anticorrelated},
    {"deep10m_1_30_float32_onesided", deep10m_1_30_float32_onesided},
    {"deep10m_1_60_float32_onesided", deep10m_1_60_float32_onesided},
    {"deep10m_1_30_float32_point", deep10m_1_30_float32_point},
    {"deep10m_1_30_float32_negation", deep10m_1_30_float32_negation},
    {"flickr_2_180_float32_real", flickr_2_180_float32_real},
    {"video-dedup_2_10000_float32_real", video_dedup_2_10000_float32_real},
};