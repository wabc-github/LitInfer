// 导入必要的包（仅保留实际使用的头文件）
#include <iostream>
#include <cstddef>  // size_t的标准头文件

// 定义模型元信息结构体（成员类型改为size_t，匹配原代码）
struct ModelMeta {
    size_t maxseq;  // 最大序列长度（无符号整型，符合size_t语义）
    size_t nkvh;    // KV头数量
    size_t dh;      // 单头维度
};

// 修正全局变量命名（避免下划线开头），初始化示例值
ModelMeta model_meta = {2048, 32, 128}; // maxseq=2048, nkvh=32, dh=128

int main() {
    // 执行原代码的核心逻辑（严格使用size_t类型）
    size_t shape_k[] = {model_meta.maxseq, model_meta.nkvh, model_meta.dh};

    // 验证数组值并解释形状含义（直观说明维度对应关系）
    std::cout << "=== shape_k数组的三维形状（Transformer KV缓存Key矩阵）===" << std::endl;
    std::cout << "第0维（序列长度维度）：shape_k[0] = " << shape_k[0] << " (maxseq)" << std::endl;
    std::cout << "第1维（KV头数量维度）：shape_k[1] = " << shape_k[1] << " (nkvh)" << std::endl;
    std::cout << "第2维（单头特征维度）：shape_k[2] = " << shape_k[2] << " (dh)" << std::endl;
    
    // 计算数组长度（size_t类型输出用%zu，C++推荐cout直接输出）
    size_t arr_length = sizeof(shape_k) / sizeof(shape_k[0]);
    std::cout << "=== 数组基本信息 ===" << std::endl;
    std::cout << "数组长度（维度数）：" << arr_length << std::endl;
    std::cout << "shape_k的完整形状：[" << shape_k[0] << ", " << shape_k[1] << ", " << shape_k[2] << "]" << std::endl;
    std::cout << "打印shape_k的类型：" << typeid(shape_k).name() << std::endl;

    return 0;
}