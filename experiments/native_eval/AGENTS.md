# Native evaluator

本模块负责可复用的原生MuJoCo评估。训练、网页和模型产物保持在各自模块。

1. 先读[README.md](README.md)确认支持范围与当前可运行命令。
2. 改核心执行、策略或Python绑定时，读[INTERFACE.md](INTERFACE.md)，再定位对应`cpp/`文件；纯C++接口以[运行时头文件](include/native_eval/runtime.hpp)为准。
3. 接轮腿模型或调整任务参数时，读[轮腿配置合同](LEGWHEEL-CONFIG.md)及[任务适配说明](docs/task-adapters.md)。新任务通过原生适配接入，核心保持任务无关。
4. 构建、导入失败或换机器时，读[构建说明](docs/build.md)。源码与构建放WSL ext4，复用已安装的MuJoCo；构建产物与依赖留在独立工作目录。
5. 改终止/重置/随机种子或评价性能时，读[验收说明](docs/validation.md)，运行受影响的验证；已完成和未覆盖的实际证据见[验证记录](VALIDATION.md)。
6. 改训练后端、实时参数同步、指标或控制台接入时，读[Brax训练接入](docs/brax-training.md)，再核对最新[短训练对照](TRAINING-INTEGRATION-RESULTS.md)。

热循环在C++内执行。Python调用方以相同接口解析配置和接收结果；指标语义、错误模式和性能变化须同步更新对应合同与验证记录。
