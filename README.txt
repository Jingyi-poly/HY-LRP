Stochastic LRP — 教授提交版

1. 内容
src-ini/：三层 SDDP/SBC/Level Set 算法、物理路线 CG 加速、独立两阶段 EF。
data/：原始 TSPLIB 坐标及预生成的 kro/berlin 随机实例。
RouteOpt-main/：C++ 定价依赖源码及原始许可文件。
本包不含项目测试、调试记录、历史实验结果、checkpoint 或本机编译产物。

2. 环境准备（一次性）
本机验证环境为 Python 3.11，macOS。需要有效的 Gurobi 许可、Gurobi C 头文件和 C++ 编译器。
在已激活的 Python 环境中，从本目录执行：
    python -m pip install -r requirements.txt
    bash src-ini/customized-subprob/s3backward/build.sh
    python src-ini/customized-subprob/s2backward/routeopt/build.py
build.sh 使用 PATH 中的 python，也可用 PYTHON_BIN 指定解释器。
RouteOpt 使用 C++20，默认 clang++，可通过 CXX 指定兼容编译器。
如未自动找到 Gurobi C 头文件，先设置：
    export GUROBI_INCLUDE_DIR=/实际安装路径/include
    export CXX=clang++
这些步骤不安装 Gurobi 许可。只安装 gurobipy 不一定包含所需 C 头文件。
移动目录或更换机器后应重新编译；RouteOpt 的验证记录绑定实际源码与编译位置。

3. 正式运行
完成环境准备后，从本目录运行一个命令：
    python src-ini/main.py
默认 kroA100：10 facilities、90 customers、T=5、S=3、seed=42，开仓决策期为 1、3、5。
实例目录名中的 T3 是原数据名称；main.py 会按默认参数生成实际 T5 数据。
默认启用 cg profile；从空 cuts/路线池开始，不读取历史 checkpoint。
Phase 1 无总时限：gap<1%，或最佳 LB/UB 连续20轮无超过1e-6的改进时停止，另有10000轮上限。
Phase 2 总时限10小时：gap<1%，或连续20轮无改进时停止；无改进退出不算收敛。
规模和算法参数仍在 src-ini/main.py 原参数区；命令行及 LRP_*/VRP_* 环境变量可以覆盖默认值。
输出自动生成到 experiments/results/。包内没有预先放入历史输出。

4. 独立 EF 对照
    python src-ini/compare_ef_sddp.py
对照规模与精度在 compare_ef_sddp.py 的 COMPARE_PARAMS 区设置。
该入口保留独立两阶段 EF；不将 EF 改成算法的三个求解层。
run_ef_only.py 也保留；其旧 --enumerate 选项依赖未附带的历史枚举参考包，请用默认 EF 求解或上述对照入口。

5. Concorde（可选的 fixed-TSP 加速）
原代码可使用 Concorde 加速已固定客户集合的 TSP。若无 Concorde，auto 模式回退到 native PCTSP；性能可能不同。
如需启用，可在安装其依赖/网络可用的环境中运行随附构建脚本：
    bash src-ini/tools/concorde/build_concorde.sh
脚本会下载 Concorde/QSopt。也可用 LRP_CONCORDE_BIN 指向已安装的 Concorde。
本提交包未分发本机 Concorde 二进制。

6. 结果口径
目标 gap 使用 (UB-LB)/LB；UB 由完整可行策略审计，LB 来自有效下界。
此前0.9862%的记录是带历史 cuts/路线池的续跑结果，不是本包从零启动的已验证用时或保证。
本包提供冷启动代码；达到目标的时间与平台、参数及求解过程有关。
第三方原始许可/源码版权声明保留，见 RouteOpt-main/LICENSE 及依赖文件。
