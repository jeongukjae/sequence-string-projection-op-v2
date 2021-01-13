workspace(name = "chatting-seq-proj-lite")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "org_tensorflow",
    sha256 = "fc6d7c57cd9427e695a38ad00fb6ecc3f623bac792dd44ad73a3f85b338b68be",
    strip_prefix = "tensorflow-8a4ffe2e1ae722cff5306778df0cfca8b7f503fe",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/8a4ffe2e1ae722cff5306778df0cfca8b7f503fe.tar.gz",
    ],
)
load("//:repo.bzl", "cc_tf_configure", "reverb_protoc_deps")
cc_tf_configure()
PROTOC_VERSION = "3.9.0"
PROTOC_SHA256 = "15e395b648a1a6dda8fd66868824a396e9d3e89bc2c8648e3b9ab9801bea5d55"
reverb_protoc_deps(version = PROTOC_VERSION, sha256 = PROTOC_SHA256)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-master",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/master.zip"],
)

# GoogleTest/GoogleMock framework. Used by most unit-tests.
http_archive(
     name = "com_google_googletest",
     urls = ["https://github.com/google/googletest/archive/master.zip"],
     strip_prefix = "googletest-master",
)

http_archive(
    name = "utf_archive",
    build_file = "@//third_party:utf.BUILD",
    sha256 = "262a902f622dcd28e05b8a4be10da0aa3899050d0be8f4a71780eed6b2ea65ca",
    urls = [
        "https://mirror.bazel.build/9fans.github.io/plan9port/unix/libutf.tgz",
        "https://9fans.github.io/plan9port/unix/libutf.tgz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name = "org_tensorflow")

#-----------------------------------------------------------------------------
# proto
#-----------------------------------------------------------------------------
# proto_library, cc_proto_library and java_proto_library rules implicitly depend
# on @com_google_protobuf//:proto, @com_google_protobuf//:cc_toolchain and
# @com_google_protobuf//:java_toolchain, respectively.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-3.8.0",
    urls = ["https://github.com/google/protobuf/archive/v3.8.0.zip"],
    sha256 = "1e622ce4b84b88b6d2cdf1db38d1a634fe2392d74f0b7b74ff98f3a51838ee53",
)
