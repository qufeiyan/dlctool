# dlctool - 死锁检测工具

## 项目概述
`dlctool` 是一个用于检测 Linux 系统中多线程程序死锁的工具。它结合了 `strace` 和 `GDB` 的功能，通过跟踪 `FUTEX_WAIT` 和 `FUTEX_WAKE` 系统调用，分析线程对互斥锁的等待和持有状态，使用 `Tarjan` 算法检测死锁环，帮助开发者快速定位和解决死锁问题。

## 功能特性
1. **进程跟踪**：支持根据进程 ID 或进程名跟踪目标进程。
2. **FUTEX 事件解析**：利用 `strace` 捕获 `FUTEX_WAIT` 和 `FUTEX_WAKE` 事件，并解析输出，获取线程等待和释放互斥锁的信息。
3. **GDB 集成**：通过 `GDB` 获取互斥锁的持有者线程 ID 以及线程堆栈信息，为死锁分析提供更详细的上下文。
4. **死锁检测**：使用 `Tarjan` 算法检测死锁环，清晰地展示死锁的线程和互斥锁关系。
5. ~~**异步编程**：采用 `asyncio` 实现异步操作，提高工具的执行效率和资源利用率。~~

## 安装与依赖
### 依赖工具
- `adb`：用于与 Android 设备通信（如果在 Android 环境中使用）, 用于**嵌入式linux**时，需要板子安装 `adbd`。
- `strace`：用于跟踪系统调用，获取 `FUTEX` 事件信息。
- `gdbserver`：用于远程调试，获取线程和互斥锁的详细信息。
- `aarch64-unknown-linux-gnu-gdb`：适用于 AArch64 架构的 GDB 调试器。

### 安装步骤
1. 确保上述依赖工具已正确安装并配置。
2. 克隆本项目仓库：
```bash
git clone https://github.com/qufeiyan/dlctool.git
cd dlctool
```

## 使用方法
### 命令行参数
```bash
python dlctool.py [-p PID | -e EXECUTABLE] [--host HOST] [--port PORT] [-s SYMBOL_FILE] [-d TOOLS_DIR] [-t TIMEOUT] [-o OUTPUT]
```
- `-p, --pid`：指定要跟踪的进程 ID。
- `-e, --executable`：指定要跟踪的进程名。
- `--host`：指定 `gdbserver` 的主机地址，默认为 `localhost`。
- `--port`：指定 `gdbserver` 的端口号，默认为 `12345`。
- `-s, --symbol-file`：指定符号文件，用于调试信息，默认为 `None`。
- `-d, --tools-dir`：指定工具目录，包含 `strace` 和 `gdbserver` 等工具，默认为 `/usr/local/bin/`。
- `-t, --timeout`：指定超时时间（秒），用于 `strace` 和 `GDB` 操作，默认为 `2`。
- `-o, --output`：指定输出文件，将线程和互斥锁信息输出到文件，默认为标准错误输出。

### 示例命令
#### 根据进程 ID 跟踪
```bash
python3 dlctool.py -p ${PID} --host $(remote ip)
```

#### 根据进程名跟踪
```bash
python3 dlctool.py -e my_program --host $(remote ip)
```

### 示例输出
```
检测到死锁环: 1
------------------------
[-------------------Deadlock--------------------] 
Thread 179218 "deadlock":
        waiting for mutex 0x412078
        Stack Trace:
        #0 0x0000ef35b02f216c in ?? from libc.so.6 
        #1 0x0000ef35b02f8c74 in pthread_mutex_lock from libc.so.6 
        #2 0x00000000004009e4 in thread2_function at deadlock.c:41 
        #3 0x0000ef35b02f595c in ?? from libc.so.6 
        #4 0x0000ef35b035ba4c in ?? from libc.so.6 
---|> held by
Thread 179217 "deadlock":
        waiting for mutex 0x4120a8
        Stack Trace:
        #0 0x0000ef35b02f216c in ?? from libc.so.6 
        #1 0x0000ef35b02f8c74 in pthread_mutex_lock from libc.so.6 
        #2 0x0000000000400970 in thread1_function at deadlock.c:19 
        #3 0x0000ef35b02f595c in ?? from libc.so.6 
        #4 0x0000ef35b035ba4c in ?? from libc.so.6 
---|> held by
Thread 179218 "deadlock":
```
> 注意：以上示例输出仅为演示目的，实际输出可能因环境和代码而异。?? 表示缺少c库符号表无法解析地址。


## 代码结构
### 主要模块
1. **`parse_arguments`**：解析命令行参数，处理用户输入。
2. **`get_pid_from_executable`**：根据进程名获取进程 ID。
3. **`WrapperStrace`**：封装 `strace` 命令的执行和输出解析，跟踪 `FUTEX` 事件。
4. **`GDBController`**：封装 `GDB` 操作，包括启动 `gdbserver`、发送命令、读取输出、获取互斥锁持有者和线程堆栈信息。
5. **`DeadLockChecker`**：使用 `Tarjan` 算法检测死锁环，格式化并输出死锁信息。

### 文件结构
```
dlctool/
├── dlctool.py           # 主程序文件
└── README.md            # 项目说明文档
```

## 开发与贡献
### 代码贡献
如果你想为该项目做出贡献，可以按照以下步骤进行：
1. 克隆项目仓库：
```bash
git clone https://github.com/qufeiyan/dlctool.git
cd dlctool
```
2. 创建并激活虚拟环境（可选）：
```bash
python3 -m venv venv
source venv/bin/activate
```
3. ~~安装依赖：~~
```bash
pip install -r requirements.txt
```
4. 进行代码修改和测试。
5. 提交你的更改：

### 问题反馈
如果你在使用过程中遇到问题或有改进建议，请在项目的 [Issues](https://github.com/qufeiyan/dlctool/issues) 页面提交问题。

## 许可证
本项目使用 [MIT 许可证](https://opensource.org/licenses/MIT)，你可以自由使用、修改和分发本项目的代码。

## 联系信息
如果你有任何问题或建议，请联系 [仓库维护者]()