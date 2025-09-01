# 数据集网站
[遇见数据集-让每个数据集都被发现，让每一次遇见都有价值。](https://www.selectdataset.com/)
[Hugging Face – The AI community building the future.](https://huggingface.co/datasets)
[首页-OpenDataLab](https://opendatalab.com/home)
[数据集 | HyperAI超神经](https://hyper.ai/cn/datasets)
[Find Open Datasets and Machine Learning Projects | Kaggle](https://www.kaggle.com/datasets)

# 大型通用数据集

	以下几个数据集都具有大规模、多模态特点，且大规模实采，可训练泛化能力
## Open X-Embodiment (OXE)

项目地址：[Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://robotics-transformer-x.github.io/)
**整合列表**：[Open X-Embodiment Dataset Overview - Google Sheets](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?pli=1&gid=0#gid=0)
	（包括：Berkeley Cable Routing 数据集、CLVR Jaco Play Dataset、RT-1 Robot Action 数据集、Language-Table 数据集、BridgeData V2 数据集、BC-Z 数据集等）
**类型**：涵盖单臂机器人、双臂机器人和四足机器人等**22种不同类型的机器**
**规模**：整合了60个已有数据集，涵盖311个场景、 100 多万条真实机器人轨迹，包括527种技能、160266项任务（当前总和约9tb）
- **数据集格式：** 所有源数据集统一转化为RLDS格式，便于下载和使用。  
    —— 该格式能够适应不同机器人设置的各种动作空间和输入模态，比如不同数量的 RGB 相机、深度相机（RGB-D相机）和点云数据。  
    —— 支持在所有主流深度学习框架中进行高效的并行数据加载。  
- **不同数据源数据处理：包括多视角处理、图像尺寸调整和动作转换等。**  
    **1）多视角处理**  
    不同数据集的视角信息差异极大。数据采集所使用的相机类型多样，包括RGB 相机与 RGBD 相机；从相机位置来看，有周视相机、腕部相机，部分数据集中所使用的相机数量甚至多达 4 部以上。  
    对于存在多视角的数据集，只会挑选其中一个被定义为“canonical” 的视角图像。这个图像通常是最接近自上而下第一人称视角的，或是与本体感知相关的，亦或是清晰度较高的那一个。  
    **2）图像尺寸调整**  
    将图像resize到320×256（width×height）。（据相关人士透露，实际上各个团队提交的数据规格并没有完全和Google保持一致。）  
    **3）动作转换**  
    将原有的动作（如关节位置）转换为末端执行器的动作，动作量可能为相对值或绝对值。在模型输出动作标记（action tokens）后，根据不同的机器人做不同的逆归一化（de-normalization）后再下达具体的控制指令。
- **数据集构成**
	1. 在使用机器人类型上，数据集中涉及的机器人包括单臂、双臂和四足机器人，其中Franka、xArm、Sawyer机器人占多数。
	2. 在场景分布上，Franka机器人占据主导地位，其次是Google Robot和xArm。
	3. 在轨迹分布上，xArm贡献了最多的轨迹数量，其次是Google Robot、Franka、 Kuka iiwa 、Sawyer和WidowX。
	4. 在数据集中常见的技能上，主要集中在Picking（抓）、Moving（移动）、Pushing（推）、Placing（放），整体呈现长尾分布，尾部有许多如Wiping、Assembling、Turning on等难度更高的技能。
	5. 在数据集常见的物品有家用电器、食品和餐具等，种类繁多。



## AgiBot World

项目地址：[AGIBOT WORLD](https://agibot-world.com/?utm_source=ai-bot.cn)
github：[GitHub - OpenDriveLab/AgiBot-World: [IROS 2025] The Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems](https://github.com/OpenDriveLab/agibot-world)
模型库：[agibot-world (AgiBot World)](https://huggingface.co/agibot-world)
- **规模**：100+机器人，100w+轨迹，alpha约9tb，beta约42tb
- **场景**：5大类100+场景
- **采集**：**遥操作单臂**；8个摄像头环绕式布局，6个主动自由度的**灵巧手**；全身最高32个自由度，末端六维力传感器和高精度视触觉传感器的配备
- 相比于 Google 开源的 Open X-Embodiment 数据集，AgiBot World 的长程数据规模高出 10 倍，场景范围覆盖面扩大 100 倍，数据质量从实验室级上升到工业级标准
- **限制**：仅使用 Agibot G1 系列同构人形机器人，无法验证泛化能力

|               | alpha  | beta    |     |
| ------------- | ------ | ------- | --- |
| task          | 36     | 217     |     |
| action slices | 185722 | 1000041 |     |
| hour          | 595.31 | 2976.4  |     |
| terabyte      | 8.74   | 41.61   |     |
[AgiBot World Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)[AgiBot World Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha)

## ARIO

项目地址：[ARIO](https://ario-dataset.github.io/)
- **规模**：ARIO数据集共有258个场景，321,064个任务，和3,033,188个演示轨迹。数据模态涵盖2D图像、3D点云、声音、文本和触觉形式。
- **来源**：数据有3大来源，一是通过布置真实环境下的场景和任务进行真人采集；二是基于MuJoCo、Habitat等仿真引擎，设计虚拟场景和物体模型，通过仿真引擎驱动机器人模型的方式生成；三是将当前已开源的具身数据集，逐个分析和处理，转换为符合ARIO格式标准的数据。
- **采集**：真人采集部分采用松灵Cobot Magic主从双臂机器人平台
- **特性**：
	- **多种感官模态**：ARIO数据集支持五种感官模态：2D图像、3D视觉、声音、文本和触觉。这种多模态支持旨在提供更丰富和多样化的数据，以增强机器人的感知和交互能力。  
	- **多模态数据的时间对齐**：ARIO支持基于时间戳的记录和命名机制，以确保多模态数据的同步。具体来说，相机数据以30Hz的频率记录，激光雷达（lidar）数据以10Hz记录，本体感觉（proprioception）数据以200Hz记录，触觉数据以100Hz记录。  
	- **统一的数据架构**：数据集采用场景-任务-片段的层次结构，每个场景和任务都有详细的文本描述。这种结构化的数据组织方式有助于系统地记录和分析数据。  
	- **统一的配置**： ARIO通过配置文件以统一格式指定数据内容，支持多种机器人形态（如单臂、双臂、仿人、四足、移动机器人）和不同的控制动作（如位置、方向、速度、扭矩等）。这种灵活的配置方式使得数据集能够适应不同类型的机器人和控制需求。  

## RoboMIND

项目地址：[RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation](https://x-humanoid-robomind.github.io/)
- **规模**：
	包含10.7万条机器人轨迹（任务成功的轨迹），涵盖479种任务、96种物体类别和38种操作技能
- **特征**
	**1. 机器人形态多样化**：
		单臂机器人Franka、人形机器人天工、双臂机器人AgileX
	**2. 任务长度多样化**：
		Franka Emika Panda机器人平均轨迹长度为179帧；UR5e机器人平均轨迹长度为158帧。这两款单臂机器人执行任务的轨迹相对较短（少于200个时间步），适合用于训练基础技能。相比之下，人形机器人“天工”（平均轨迹长度为669帧）和双臂机器人AgileX Cobot Magic V2.0（平均轨迹长度为655帧）执行任务的轨迹相对较长（超过500个时间步），更适合用于长时间跨度的任务训练以及技能组合
	**3. 任务多样性**
		- **铰链式操作**（Artic. M. - Articulated Manipulations ）：开关抽屉、门，转动物体等。
		- **协调操作**（Coord. M. - Coordination Manipulations）：这里主要是指双臂机器人的协同。
		- **基本操作**（Basic M. - Basic Manipulations）：抓、握、提、放等最基本的操作。
		- **多物体交互**（Obj. Int.- Multiple Object Interactions）：比如推一个物体撞另一个物体。
		- **精准操作**（Precision M. - Precision Manipulations）：比如倒水，穿针引线这种需要精确控制的操作。
		- **场景理解**（Scene U. - Scene Understanding）：比如从特定位置关门，或者把不同颜色的积木放到对应颜色的盒子里。
## RH20T

项目地址：[RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot](https://rh20t.github.io/)

- **规模**：
	数据集总数据量达20TB，包含超 11 万个高接触度机器人操作序列与等量的11万个人类演示视频，共计超 5000 万帧图像。该数据集包含视觉、触觉、音频等多模态信息，覆盖147种任务（如切割、折叠、旋转等接触密集型操作）与42种技能（涵盖抓取、放置、装配等技能），从日常基础操作到复杂技能均有涉及。平均每项技能包括 750 次机器人操作，为机器人学习与技能优化提供了丰富的实践样本
- **采集**：
	用4种主流机械臂（UR5、Franka、Kuka和Flexiv）、4种夹爪（Dahuan  AG95、WSG50、Robotiq-85和Franka）、3种力传感器（OptoForce 、ATI Axia80-M20和Franka），共7种机器人硬件配置组合。
- **多模态**：
	- 视觉信息：RGB图像、深度图像及双目红外图像三种相机的视觉信息；
	- 触觉信息：提供机器人腕部的六自由度力/扭矩测量，部分序列还包含指尖触觉信息；
	- 音频信息：包括夹爪内部与全局环境的声音记录；
	- 本体感知信息：涵盖关节角度/扭矩、末端执行器的笛卡尔位姿及夹爪状态。
- **树状层级结构组织**
	- 数据集以任务内部相似性为基础构建树状结构，节点层级反映任务的抽象程度。如下图所示，根节点代表最广泛的任务类别，随着层级下移，节点逐渐细分到具体任务。叶子节点代表最细粒度的任务实例，具有最近共同祖先的叶子节点在语义和执行方式上更相似
	- 每个任务通过组合不同层级的叶子节点生成数百万个<人类演示，机器人操作> 数据对
- **机器人操作数据与人类演示视频数据配对**
	RH20T数据集通过树状结构组织之间的任务相似性，并利用跨层级叶节点配对构建密集的多样化数据对，旨在解决机器人操作中视角、场景、硬件差异带来的泛化挑战。这种设计为训练通用型机器人基础模型提供了结构化支持。
	- **多模态配对**：一个机器人操作序列（红色叶节点）可与多个不同视角、场景、操作者的人类示范视频（绿色叶节点）配对。
	- **跨层级配对**：通过选择不同层级的共同祖先，可生成数百万对<人类示范，机器人操作>数据。共同祖先越近，叶节点关联性越强（例如“插拔USB-A”与“插拔USB-C”比“插拔插座”更相似）。




# 中小数据集

## BridgeData V2（语言动作基础）

项目地址：[BridgeData V2](https://rail-berkeley.github.io/bridgedata/)
Each trajectory is labeled with a natural langauge instruction corresponding to the task the robot is performing.
- **60,096 轨迹（小规模）**
    - 50,365 teleoperated demonstrations（大部分为真实遥控收集）
    - 9,731 rollouts from a scripted pick-and-place policy（部分脚本生成动作）
- **24 环境**（4大类24种环境，其中厨房环境占大部分）
- **13 技能**（包括捡起和放置、推、擦、堆叠、折叠等）
- The control frequency is 5 Hz and the average trajectory length is 38 timesteps.
- **采集**：1个顶部固定深度相机，2个位置随机相机，1个固定机械臂的第一视角相机

## RoboSet

项目地址：[RoboSet Dataset](https://robopen.github.io/roboset/)
- **规模**：真实采集共28,500轨迹，其中9500 遥控, 19000 动觉演示；外加70000脚本生成轨迹
- **采集**：4相机
- 主要围绕日常厨房活动场景开展，例如泡茶、烘焙等场景。
## BC-Z

项目地址：[bc-z](https://sites.google.com/view/bc-z/)
- **规模**：25,877个操作任务，100种多样化的任务
- **特性**：**单场景**；结合了专家演示和人类纠错；
- 项目目的是通过模仿学习实现对新任务的**零样本泛化**，对应的特性是**任务多样性、语义标注**
## DROID（Distributed Robot Interaction Dataset）

项目地址：[DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset](https://droid-dataset.github.io/)
- **规模**：具有 76k 条演示轨迹或 350 小时的交互数据， 564 个场景和 86 个任务，共1.7TB数据（相对于OXE，**轨迹更少但是场景更多**）
- **设备**：The setup consists of a Franka Panda 7DoF robot arm, two adjustable Zed 2 stereo cameras, a wristmounted Zed Mini stereo camera, and an Oculus Quest 2 headset with controllers for teleoperation
- **优点**：**多种动词长尾**


## CALVIN

**项目地址**：[GitHub - mees/calvin: CALVIN - A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://github.com/mees/calvin)
**特点**：
- 100+小时的模拟机器人操作数据
- 包含**详细语言指令**与动作对齐，专注于**语言指令驱动**的机械臂操作
- 桌面任务：抓取、放置、排序、组合等
- 提供状态观测和RGB图像


## bytedance_robot_benchmark_20

项目地址：[RoboVLMs: What Matters in Building Vision-Language-Action Models for Generalist Robot Policies](https://robovlms.github.io/)
- VLA数据集，包含8K高质量轨迹，20个常见操作任务

## Franka Kitchen


项目地址：[Franka Kitchen - Gymnasium-Robotics Documentation](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/)
- **模拟环境**：MuJoCo
- 专注于**厨房桌面操作**：开门、按按钮、移动物体等
- 每个任务有明确的子目标
- 复杂多步骤桌面操作任务、任务链学习

## BEHAVIOR-1K benchmark（仿真）

项目地址：[BEHAVIOR - BEHAVIOR](https://behavior.stanford.edu/index.html)
- **规模**：包括1000个日常活动，50多个场景，用物理和语义属性标注的5000个物体
- **模拟环境**：OmniGibson，基于NVIDIA's IsaacSim（Omniverse平台的仿真套件），专注日常环境
- 任务长程，难度高；
- 仿真套件提供了较多素材，自定义


## ALFRED benchmark（仿真）

项目地址：[ALFRED -- A Benchmark for Interpreting Grounded Instructions for Everyday Tasks](https://askforalfred.com/)
- **规模**：包含 25,743 条自然语言指令，涵盖 8,055 个专家演示，平均每个任务包含 50 个动作步骤，生成 428,322 个图像-动作对；每个专家演示都可以在 AI2-THOR 2.0 模拟器中确定性地重播
- **特性**：与常规视觉语言导航数据集不同，ALFRED除了目标指令外，还包含了每一个子目标的分步语言指令作为指导；指令分为高层次目标（如“将杯子放入咖啡机”）和低层次步骤（如“走向右侧的咖啡机”）

- 文件结构：[alfred/data at master · askforalfred/alfred · GitHub](https://github.com/askforalfred/alfred/tree/master/data)
	Expert Demonstration:
## LIBERO benchmark（仿真）

项目地址：[LIBERO Datasets – LIBERO](https://libero-project.github.io/datasets)
- **环境**：基于 MuJoCo和robosuite
- **数据内容**：每个演示轨迹包含一系列时间步的数据，每个时间步通常包括：
	- **观测 (Observation)**:
	    - `agent_view`: 机器人主视角的 RGB 图像，通常分辨率为 256x256 像素 .
	    - `state` 或 `robot_state`: 机器人末端执行器（End-Effector）的状态，如 6D 位姿（位置和方向）和夹爪状态 .
	- **动作 (Action)**: 机器人应执行的动作，通常是连续的控制指令，例如末端执行器的位移（Δx, Δy, Δz）和旋转（Δroll, Δpitch, Δyaw），以及夹爪的开合指令 .
	- **任务描述 (Task Description)**: 用自然语言描述的当前任务目标，例如“把红色的方块放进蓝色的碗里”
- **子数据集**：
	1. **`libero_spatial`**: 评估模型对**空间关系**的理解和泛化能力，例如“将物体放在杯子的左边” .
	2. **`libero_object`**: 评估模型对**不同物体**的泛化能力，测试模型能否将在一种物体上学到的技能迁移到新物体上 .
	3. **`libero_goal`**: 评估模型对**不同目标状态**的泛化能力，例如根据一张目标图像来完成任务 .
	4. **`libero_100`**: 这是一个包含100个任务的大型套件，涵盖了更广泛的操作技能，是评估综合性能的主要基准 . 该套件有时也被拆分为 `libero_90` 和 `libero_10` .
- **架构**：
	- **资源层**：通过 `assets` 和 `scenes` 管理基础模型。
	- **定义层**：通过 Python 类程序化地定义和初始化场景。
	- **目标层**：通过 BDDL 文件声明式地定义任务成功条件。
	- **交互层**：遵循标准的 Gym 接口与学习算法交互。
	- **数据层**：提供并支持创建标准化的 HDF5 演示数据集。
- 


## RLBench

项目地址：[RLBench](https://sites.google.com/view/rlbench)
- **环境**：CoppeliaSim v4.1.0 and [PyRep](https://github.com/stepjam/PyRep)，兼容gym
- **内容**：
	- 包含**100+桌面操作任务**
	- 每个任务提供**专家演示轨迹**（通过IK控制器生成）
	- 支持RGB、深度、分割图等多种视觉输入
- **特点**：
	- 可直接用于RL，IL训练
	- 可自定义任务


## ManiSkill2

项目地址：[ManiSkill](https://www.maniskill.ai/)
- **环境**：基于SAPIEN 2.0，物理模拟精确
- **内容**：专注于**精细操作技能**，特别**适合桌面任务**
	- 插销任务（Peg Insertion）
	- 门把手操作（Door Opening）
	- 拼图任务（Puzzle）
	- 拧盖子（Jar Cap）
	- 按键操作（Button Press）
- 提供**状态观测**和**视觉观测**两种模式

## YCB-Video

项目地址：[PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes – UW Robotics and State Estimation Lab](https://rse-lab.cs.washington.edu/projects/posecnn/)
- 主要用于**6D对象姿态估计**训练
- **内容**：数据集提供了来自YCB数据集的21个对象的精确6D姿势，这些对象出现在92个视频中，整个数据集包含133,827帧
- 视频帧经过精细标注，包括物体边界框、3D模型对齐信息以及深度图
- 适用于多种计算机视觉任务，如物体检测、姿态估计和场景理解

## RoboTurk

项目地址：[RoboTurk - Crowdsourcing Robotics](https://roboturk.stanford.edu/)
- 通过众包收集的大规模机器人操作数据集
- **规模**：1,000+小时操作数据，15,000+任务实例
- 适合研究人类偏好学习
