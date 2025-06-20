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

DataCard sift_1_500_int32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    500,
    "int32",
};

DataCard sift_1_1000_int32{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1'000'000,
    10'000,
    100,
    1,
    1000,
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

DataCard gist_1_500_int32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    500,
    "int32",
};

DataCard gist_1_1000_int32{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    1'000'000,
    1'000,
    100,
    1,
    1000,
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

DataCard crawl_1_500_int32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    500,
    "int32",
};

DataCard crawl_1_1000_int32{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10'000,
    100,
    1,
    1000,
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

DataCard glove100_1_500_int32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    500,
    "int32",
};

DataCard glove100_1_1000_int32{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10'000,
    100,
    1,
    1000,
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

// DataCard deep10m_1_10000_float32{
//     "crawl",
//     crawl_bpath,
//     crawl_qpath,
//     crawl_gpath,
//     300,
//     1'989'995,
//     10'000,
//     100,
//     1,
//     10000,
//     "float32",
// };

std::map<std::string, DataCard> name_to_card{
    {"siftsmall_1_10_int32", siftsmall_1_10_int32},
    {"siftsmall_1_1000_int32", siftsmall_1_1000_int32},
    // {"siftsmall_1_100_float32", siftsmall_1_100_float32},
    {"siftsmall_1_1000_float32", siftsmall_1_1000_float32},
    {"siftsmall_1_1000_top500_float32", siftsmall_1_1000_top500_float32},
    {"siftsmall_2_1000_float32", siftsmall_2_1000_float32},
    {"sift_1_2_int32", sift_1_2_int32},
    {"sift_1_5_int32", sift_1_5_int32},
    {"sift_1_10_int32", sift_1_10_int32},
    {"sift_1_50_int32", sift_1_50_int32},
    {"sift_1_100_int32", sift_1_100_int32},
    {"sift_1_500_int32", sift_1_500_int32},
    {"sift_1_1000_int32", sift_1_1000_int32},
    {"sift_1_10000_float32", sift_1_10000_float32},
    {"sift_2_10000_float32", sift_2_10000_float32},
    {"sift_3_10000_float32", sift_3_10000_float32},
    {"sift_4_10000_float32", sift_4_10000_float32},
    {"gist_1_2_int32", gist_1_2_int32},
    {"gist_1_5_int32", gist_1_5_int32},
    {"gist_1_10_int32", gist_1_10_int32},
    {"gist_1_50_int32", gist_1_50_int32},
    {"gist_1_100_int32", gist_1_100_int32},
    {"gist_1_500_int32", gist_1_500_int32},
    {"gist_1_1000_int32", gist_1_1000_int32},
    {"gist_1_10000_float32", gist_1_10000_float32},
    {"gist_2_10000_float32", gist_2_10000_float32},
    {"gist_3_10000_float32", gist_3_10000_float32},
    {"gist_4_10000_float32", gist_4_10000_float32},
    {"crawl_1_2_int32", crawl_1_2_int32},
    {"crawl_1_5_int32", crawl_1_5_int32},
    {"crawl_1_10_int32", crawl_1_10_int32},
    {"crawl_1_50_int32", crawl_1_50_int32},
    {"crawl_1_100_int32", crawl_1_100_int32},
    {"crawl_1_500_int32", crawl_1_500_int32},
    {"crawl_1_1000_int32", crawl_1_1000_int32},
    {"crawl_1_10000_float32", crawl_1_10000_float32},
    {"crawl_2_10000_float32", crawl_2_10000_float32},
    {"crawl_3_10000_float32", crawl_3_10000_float32},
    {"crawl_4_10000_float32", crawl_4_10000_float32},
    {"glove100_1_2_int32", glove100_1_2_int32},
    {"glove100_1_5_int32", glove100_1_5_int32},
    {"glove100_1_10_int32", glove100_1_10_int32},
    {"glove100_1_50_int32", glove100_1_50_int32},
    {"glove100_1_100_int32", glove100_1_100_int32},
    {"glove100_1_500_int32", glove100_1_500_int32},
    {"glove100_1_1000_int32", glove100_1_1000_int32},
    {"glove100_1_10000_float32", glove100_1_10000_float32},
    {"glove100_2_10000_float32", glove100_2_10000_float32},
    {"glove100_3_10000_float32", glove100_3_10000_float32},
    {"glove100_4_10000_float32", glove100_4_10000_float32},
    {"audio_1_10000_float32", audio_1_10000_float32},
    {"audio_2_10000_float32", audio_2_10000_float32},
    {"audio_3_10000_float32", audio_3_10000_float32},
    {"audio_4_10000_float32", audio_4_10000_float32},
    {"video_1_10000_float32", video_1_10000_float32},
    {"video_2_10000_float32", video_2_10000_float32},
    {"video_3_10000_float32", video_3_10000_float32},
    {"video_4_10000_float32", video_4_10000_float32},
};