import argparse
from collections import defaultdict
import io
import json
from multiprocessing import Queue
from queue import Empty
import select
import sys
import subprocess
import fcntl
import os
import re
from threading import Thread
import time
from typing import Dict, List, Optional, Set, Tuple

# 打印带有颜色的文本
# 参数:
# data: 要打印的文本
# color: 文本的颜色，默认为蓝色
def print_color(data, color='blue'):
    # 定义颜色代码字典
    color_codes = {
        'blue': '\033[94m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'pink': '\033[95m',
        'end': '\033[0m'
    }
    # 检查指定的颜色是否在支持的颜色列表中
    if color in color_codes:
        # 获取对应的颜色代码
        color_code = color_codes[color]
        end_code = color_codes['end']
        # 打印带有颜色的文本
        print(f"{color_code}{data}{end_code}")
    else:
        # 如果指定的颜色不支持，则直接打印文本
        print(data)

def get_pid_from_executable(name: str) -> Optional[str]:
    """根据进程名获取进程ID"""
    pid :Optional[str] = None
    cmd = [ 'adb', 'shell', 'pidof', name]
    try:
        pid = subprocess.check_output(cmd).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        sys.stderr.write(f"Error: process '{name}' not found.\n")
        sys.exit(1)
    except FileNotFoundError:
        sys.stderr.write("Error: 'pidof' command not found.\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error getting PID: {e}\n")
        sys.exit(1)
    return pid  

def set_noblock(fd):
    # 设置非阻塞读取
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

def read_lines_with_timeout(file: io.IOBase, timeout: float) -> List[str]:
    """
    按行从 IO 对象读取数据，设置超时时间。

    :param file: IO 对象，如文件对象或标准输入输出对象
    :param timeout: 超时时间（秒）
    :return: 包含读取到的行的列表，如果超时则返回空列表
    """
    lines = []
    try:
        # 获取 IO 对象的文件描述符
        fd = file.fileno()
        start_time = time.time()
        while time.time() - start_time < timeout:
            # 使用 select 监控文件描述符的可读状态
            ready, _, _ = select.select([fd], [], [], 0.1)
            if fd in ready:
                # 读取一行数据
                line = file.readline()
                if line:
                    lines.append(line.strip())
    except Exception as e:
        sys.stderr.write(f"Error reading lines: {e}\n")
    return lines

class WrapperStrace(object):
    """
    run strace -e futex -f -p pid
    parse output
    """
    def __init__(self, pid: int, strace_path: str="./strace", timeout: int = 4):
        self.pid = pid
        self.timeout = timeout
        self.strace_path = strace_path

    def run(self) -> Set[Tuple[str, str, str]]:
        strace_path = f'{self.strace_path}'
        strace_cmd = ['adb', 'shell', 'sudo', strace_path, '-e', 'futex', '-f', '-p', str(self.pid)]
        try:
            proc = subprocess.Popen(
                strace_cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1, # 设置缓冲区大小为1，以便实时读取输出
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            is_timeout = True
            proc.kill() # 超时后，returncode 为 -9
            stdout, stderr = proc.communicate()
        except FileNotFoundError:
            sys.stderr.write("Error: 'strace' command not found.\n")
            sys.stderr.write("help: You must specify a path to tools.\n")
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"Error starting strace: {e}\n")
            sys.exit(1)

        # 只能检查adb 的 returncode, strace 错误输出经过adb也成了标准输出 
        if proc.returncode != 0 and not is_timeout:
            if "Operation not permitted" in stdout:
                sys.stderr.write(f"rc:{proc.returncode} 权限不足：尝试使用 sudo 或以 root 身份运行脚本")
            else:
                sys.stderr.write(f"rc:{proc.returncode} strace 错误：{stderr.strip()}")
            sys.exit(1) 
        res = set()
        
        # 正则表达式匹配futex WAIT事件
        pattern = re.compile(
            r'\[pid\s+(\d+)\] futex\((0x[0-9a-f]+), (FUTEX_WAIT|FUTEX_WAKE(?:_[A-Z]+)?\b)'
        )
        for line in stdout.splitlines():
            match = pattern.match(line)
            if match:
                tid = match.group(1)
                mutex = match.group(2)
                status = match.group(3)
                # print (f"tid:{tid} mutex:{mutex} status:{status}")
                if "FUTEX_WAIT" in status:
                    res.add((tid, mutex, "wait"))
                elif "FUTEX_WAKE" in status:
                    # when you are not sure if the element exists in the set. It does not raise any exception if the element is not found.
                    res.discard((tid, mutex, "wait"))
            elif "not found" in line or "Operation not permitted" in line:
                advice = "If Operation not permitted, You must restart the adbd in root mode."
                sys.stderr.write(f"strace 输出：{stdout}\n advice:{advice}\n")
                sys.exit(1)
        return res

class GDBController:
    def __init__(self, pid: int, host, port=12345, symbol_file:str|None=None, gdbserver_path="/usr/local/gdbserver"):
        self.host = host
        self.port = port
        self.pid = pid
        self.gdbserver_path = gdbserver_path
        
        gdb_cmd = ['aarch64-unknown-linux-gnu-gdb', '--quiet', '--nx', '-i=mi', '--se', f'{symbol_file}' ] 
        config = [ 
            "-ex", "set pagination off",
            "-ex", "set logging file gdb.txt",
            "-ex", "set logging enabled on"
        ]
        gdb_cmd.extend(config)
        # print(' '.join(gdb_cmd))
        try:
            self._gdb = subprocess.Popen(
                gdb_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
        except FileNotFoundError as e:
            sys.stderr.write(f"Error starting gdb: {e}\n")
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"Error starting gdb: {e}\n")
            sys.exit(1)

        self._response_queue = Queue()
        self._reader_thread = Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

        # 初始化GDB会话
        try: 
            res = self._wait_for_ready()
            success, msg = self._start_server()
            if not success:
                sys.stderr.write(f"Error starting gdbserver: {msg}\n")
                sys.exit(1)
            self._send_command(f"target remote {host}:{port}")
            res = self._wait_for_ready(15) # 远程连接超时10s, 才会输出信息，这里设置15s以捕获输出
            if "Operation timed out" in res:
                sys.stderr.write(f"Error connecting to {host}:{port}\n")
                sys.exit(1)
            self._send_command("set solib-search-path ./")
            self._wait_for_ready()
        except RuntimeError as e:
            sys.stderr.write(f"Error initializing gdb: {e}\n")
            sys.stderr.write(self._gdb.stderr.read())
            sys.exit(1)

    def _start_server(self, timeout: float=3.0) -> Tuple[bool, str]:
        """
        通过 adb 启动 gdbserver 并检测权限问题
        :param pid: 目标进程的 PID
        :param port: 调试端口 (默认 12345)
        :param timeout: 命令执行超时时间 (秒)
        :return: (是否成功, 输出信息/错误信息)
        """
        # 构建启动GDBServer的命令列表
        cmd = [
            'adb', 'shell', f'{self.gdbserver_path}', f':{self.port}', "--attach", f'{self.pid}'
        ]
        try:
            # 使用subprocess.Popen启动GDBServer进程
            proc = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,  # 将标准输出重定向到PIPE
                stderr=subprocess.PIPE,  # 将标准错误也重定向到标准输出
                text=True,  # 以文本模式读取输出
                bufsize=1,
            )
            self._server = proc
        except subprocess.CalledProcessError as e:
            # 捕获并处理subprocess.Popen执行过程中的异常
            sys.stderr.write(f"GDBServer error for {self.pid}: {e.output}\n")  # 将错误信息输出到标准错误
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"Error starting gdb: {e}\n")
            sys.exit(1)
            # 合并 stdout 和 stderr
        output = '\n'.join(read_lines_with_timeout(proc.stdout, timeout))
        # 权限错误检测
        permission_errors = [
            r"permission\s+denied",   # 匹配 "permission denied"
            r"Operation\s+not\s+permitted",  # EPERM 错误
            r"cannot\s+open\s+/proc/",  # 常见于无权限访问进程
            r"access\s+denied"        # 通用拒绝访问
        ]
        # 检查输出中是否包含权限错误
        if any(re.search(pattern, output, re.IGNORECASE) for pattern in permission_errors):
            advice = (
                "\n[解决方案建议]\n"
                "1. 确认设备已 root 并授予 adb root 权限\n"
                "2. 使用调试版 ROM 或关闭 SELinux:\n"
                "   adb shell setenforce 0\n"
                "3. 对非 root 设备尝试使用 run-as 命令:\n"
                "   adb shell run-as <package> gdbserver :{port} --attach {pid}"
            ).format(port=self.port, pid=self.pid)
            proc.kill()
            return (False, f"权限不足: {output.strip()}\n{advice}")
            # 其他错误检测
            
        # 成功检测
        if "Listening on port" in output:
            return (True, f"gdbserver 已启动: {output.strip()}")
        proc.kill()
        return (False, f"未知错误: {' '.join(cmd)}\n{output.strip()}\n")
       

    def _send_command(self, command):
        """发送命令到GDB"""
        self._gdb.stdin.write(f"{command}\n")
        self._gdb.stdin.flush()
    
    def _read_output(self):
        """持续读取GDB输出"""
        while True:
            line = self._gdb.stdout.readline()
            if not line:
                break
            self._response_queue.put(line)

    def _wait_for_ready(self, timeout=5):
        """等待GDB准备就绪"""
        end_marker = '(gdb)'
        response = []
        while True:
            try:
                line = self._response_queue.get(timeout=timeout)
                response.append(line)
                if end_marker in line:
                    return ''.join(response)
            except Empty:
                raise RuntimeError("GDB响应超时!!!")

    def get_owner(self, mutex_addr) -> str|None:
        """查询mutex持有者线程ID"""
        cmd = f"print ((pthread_mutex_t*){mutex_addr})->__data.__owner"
        self._send_command(cmd)
        try:
            response = self._wait_for_ready()
        except RuntimeError as e:
            sys.stderr.write(f"Error querying GDB: {e}\n")
            sys.exit(1)
        # print(response)
        # 解析输出结果
        match = re.search(
            r'\$[0-9]+\s+=\s+([0-9\-]+)',
            # r'\$(\d+)\s*=\s*(-?\d+)',
            response,
            re.MULTILINE
        )
        return match.group(1) if match else None

    def close(self):
        """关闭GDB连接"""
        self._send_command('quit')
        self._gdb.wait()
        self._server.wait()

    def _parse_thread_stack(self, gdb_output: str) -> list:
        """
        解析GDB堆栈输出为结构化数据
        :param gdb_output: 原始GDB输出字符串
        :return: 排序后的堆栈列表（按层级降序）
        """
        # 清洗数据并转换为标准JSON
        json_str = gdb_output.replace("frame=", "").replace("arch=", "\"arch\":")
        json_str = json_str.replace("level=", "\"level\":") \
                        .replace("addr=", "\"addr\":") \
                        .replace("func=", "\"func\":") \
                        .replace("file=", "\"file\":") \
                        .replace("fullname=", "\"fullname\":") \
                        .replace("line=", "\"line\":") \
                        .replace("from=", "\"from\":") 
        json_str = "{" + json_str.split("{", 1)[-1].rsplit("}", 1)[0] + "}"
        # 处理中文编码问题        
        def transcode(matches):
            oct_list = [int(match.replace('\\', ''), 8) for match in matches]
            oct_bytes = bytes(oct_list)
            # print(f"oct_bytes: {oct_bytes}")
            decode_str = oct_bytes.decode('utf-8')
            return decode_str

        pattern = r'(\\\d{3})'
        matchs = re.findall(pattern, json_str)
        if matchs:
            decode_str = transcode(matchs)
            json_str = json_str.replace(''.join(matchs), decode_str)

        data = json.loads(f"[{json_str}]")
        return data

    def _get_thread_info(self, tids: List[str]) -> Dict[str, Dict]:
        res = defaultdict(dict)
        """mi 模式获取线程信息"""
        thread_info_cmd = "-thread-info"
        self._send_command(thread_info_cmd)
        try:
            response = self._wait_for_ready()
        except RuntimeError as e:
            sys.stderr.write(f"Error querying GDB for thread info: {e}\n")
            sys.exit(1)
        
        # 查找对应线程 ID
        thread_pattern = r'id="(\d+)",target-id="Thread \d+\.(\d+)",name="([^"]+)"'
        matches = re.findall(thread_pattern, response)
        target_thread_ids = {}
        for match in matches:
            thread_id, current_lwp_id, thread_name = match
            if current_lwp_id in tids:
                target_thread_ids[current_lwp_id] = (thread_id, thread_name)
        for lwp in target_thread_ids:
            # 切换到目标线程
            switch_thread_cmd = f"-thread-select {target_thread_ids[lwp][0]}"
            self._send_command(switch_thread_cmd)
            
            # 执行回溯命令
            backtrace_cmd = "-stack-list-frames"
            self._send_command(backtrace_cmd)
            try:
                self._wait_for_ready()
                response = self._wait_for_ready()
            except RuntimeError as e:
                sys.stderr.write(f"Error querying GDB for backtrace: {e}\n")
                sys.exit(1)
            # 解析回溯信息
            # print(f"res: {response}")
            if "stack" not in response:
                raise ValueError("No stack information found in GDB output")
            # 预处理，将其转换为接近 JSON 的格式
            try:
                stack_frames = self._parse_thread_stack(response)
                # print(f"stack_frames: {stack_frames}")  
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")   
            thread_info = {}
            thread_info['name'] = target_thread_ids[lwp][1]
            frames = []
            for frame in stack_frames:
                source = f"{frame['fullname'] if 'fullname' in frame else frame['file'] if 'file' in frame else ''}{':'+ frame['line'] if 'line' in frame else ''}"
                lib = f"{frame['from'] if 'from' in frame else '??'}"
                frames.append(f"#{frame['level']} {frame['addr']} in {frame['func']} {'at ' + source if source != '' else 'from ' + lib} ")
            thread_info['frames'] = frames
            res[lwp] = thread_info
        return res
    
    def get_thread_stack_frame(self, tids: List[str]):
        """查询mutex持有者线程ID"""
        thread_frames = self._get_thread_info(tids)
        return thread_frames

class DeadLockChecker:   
    def __init__(self, data: List[Tuple[str, str, str]], stack_info: Dict[str, Dict[str, List[str]]]):
        self.data = data
        # wait_info: {key: tid, value: mutex address}
        # held_info: {key: mutex address, value: tid}
        self.graph, self.wait_info, self.held_info = self._build_graph()
        self.stack_info = stack_info

    # Tarjan 算法查找强连通分量
    # 参数:
    # graph: 一个字典，键为节点（线程 ID 或互斥锁地址），值为该节点的邻居节点列表
    # 返回值:
    # 一个列表，列表中的每个元素是一个列表，表示一个强连通分量
    def _tarjan(self) -> List[List[int]]:
        graph = self.graph
        # 用于记录节点的索引计数器
        index_counter: List[int] = [0]
        # 用于存储栈的列表
        stack: List[str] = []
        # 用于记录每个节点的 lowlink 值的字典
        lowlink: Dict[str, int] = {}
        # 用于记录每个节点的索引的字典
        index: Dict[str, int] = {}
        # 用于存储强连通分量的列表
        result: List[List[str]] = []

        # 递归函数，用于查找强连通分量
        def strongconnect(node: str) -> None:
            # 为节点分配索引
            index[node] = index_counter[0]
            # 初始化节点的 lowlink 值
            lowlink[node] = index_counter[0]
            # 索引计数器加 1
            index_counter[0] += 1
            # 将节点压入栈中
            stack.append(node)

            # 如果节点有邻居节点
            if node in graph:
                for neighbor in graph[node]:
                    # 如果邻居节点还没有被访问过
                    if neighbor not in index:
                        # 递归调用 strongconnect 函数
                        strongconnect(neighbor)
                        # 更新当前节点的 lowlink 值
                        lowlink[node] = min(lowlink[node], lowlink[neighbor])
                    # 如果邻居节点在栈中
                    elif neighbor in stack:
                        # 更新当前节点的 lowlink 值
                        lowlink[node] = min(lowlink[node], index[neighbor])

            # 如果当前节点的 lowlink 值等于其索引值
            if lowlink[node] == index[node]:
                # 用于存储当前强连通分量的列表
                connected_component: List[str] = []
                while True:
                    # 从栈中弹出节点
                    neighbor = stack.pop()
                    # 将节点添加到当前强连通分量中
                    connected_component.append(neighbor)
                    if neighbor == node:
                        break
                # 如果强连通分量的长度大于 1
                if len(connected_component) > 1:
                    # 将强连通分量添加到结果列表中
                    result.append(connected_component)

        # 遍历图中的所有节点
        for node in graph:
            # 如果节点还没有被访问过
            if node not in index:
                # 调用 strongconnect 函数
                strongconnect(node)
        # 打印找到的强连通分量
        print(f"sccs: {result}")
        return result

    def _build_graph(self) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
        data = self.data
        # 用于存储图的字典，默认值为一个空列表
        graph: Dict[str, List[str]] = defaultdict(list)
        # 用于存储每个互斥锁的持有者的字典
        holding_mutex: Dict[str, str] = {}
        # 用于存储等待信息的字典，键为等待的线程 ID，值为该线程等待的互斥锁地址
        wait_info: Dict[str, str] = {}

        # 找出每个互斥锁的持有者
        for tid, mutex, status in data:
            if status == 'held':
                holding_mutex[mutex] = tid

        # 构建等待图并记录等待的 mutex,  graph 中 node 可以是 tid, 也可以是 mutex
        for tid, mutex, status in data:
            if status == 'wait' and mutex in holding_mutex:
                # 获取互斥锁的持有者
                holder = holding_mutex[mutex]
                # 添加从等待线程到互斥锁的边
                graph[tid].append(mutex)
                # 添加从互斥锁到持有者线程的边
                graph[mutex].append(holder)
                # 记录等待信息
                wait_info[tid] = mutex
        return graph, wait_info, holding_mutex

    def _format_deadlock_cycles(self, sccs: List[List[str]]) -> List[str]:
        # 用于存储格式化后的死锁环的列表
        formatted_cycles: List[str] = []
        # 遍历每个强连通分量
        for scc in sccs:
            cycle_str = self._cycle_str(scc) 
            # 将格式化后的死锁环添加到结果列表中
            formatted_cycles.append(''.join(cycle_str))
        return formatted_cycles
    def _cycle_str(self, scc: List[str]) -> List[str]:
        cycle_str = ["[-------------------Deadlock--------------------] \n"]

        for i in range(len(scc)):
            # 获取当前节点
            current = scc[i]
            # 判断当前元素是 tid 还是 mutex
            if current in self.wait_info:  # 如果当前是 tid
                break
        stack: List[str] = []
        def _format_cycle(current: str):
            # print(f"current: {current}")
            if current in self.wait_info: # 如果当前是 tid
                tid = current
            elif current in self.held_info: # 如果当前是 mutex
                tid = self.held_info[current]
            if tid not in stack:
                cycle_str.append(f"Thread {tid} \"{self.stack_info[tid]['name']}\":")
                cycle_str.append(f"\n\twaiting for mutex {self.wait_info[tid]}")
                cycle_str.append("\n\tStack Trace:\n\t")
                cycle_str.append("\n\t".join(self.stack_info[tid]['frames']))
                cycle_str.append("\n---|> held by\n")
                stack.append(current)
            else:
                root = tid 
                cycle_str.append(f"\tThread {root} {self.stack_info[root]['name']}:\n")
                return 
            _format_cycle(self.wait_info[tid])
        _format_cycle(current)
        return cycle_str

    def run(self):
        sccs = self._tarjan()
        if sccs:
            print_color(f"检测到死锁环: {len(sccs)}", color='red')
            formatted_cycles = self._format_deadlock_cycles(sccs)
            for cycle in formatted_cycles:
                print("------------------------")
                print_color(cycle, color='pink')
        else:
            print_color("未检测到死锁环", color='green')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Track FUTEX_WAIT events in strace output and detect deadlocks with GDB help'
    )
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-p', '--pid', type=int, help='Attach to process ID')
    parser.add_argument('-e', '--executable', type=str, help='Attach to process name')
    parser.add_argument('--host', type=str, default='localhost', help='gdbserver host address')
    parser.add_argument('--port', type=int, default=12345, help='gdbserver port')
    parser.add_argument('-s', '--symbol-file', type=str, default=None, help='Use FILE as symbol file')
    parser.add_argument('-d', '--tools-dir', type=str, default="/usr/local/bin/", help='Path to tools')
    # parser.add_argument('command', nargs=argparse.REMAINDER, type=str, help='Command to run under strace')
    parser.add_argument('-t', '--timeout', type=float, default=2,
                        help='Timeout in seconds for no output')
    parser.add_argument('-o', '--output', help='Output file (default: stderr)')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    if args.pid is None and args.executable is None:
        sys.stderr.write("Error: You must specify a process ID or name to run.\n")
        sys.exit(1)

    pid = args.pid
    if args.pid is None:
        pid = get_pid_from_executable(args.executable)
        if not pid:
            sys.stderr.write(f"Error: Could not find process with executable name {args.executable}\n")
            sys.exit(1)

    print(f"tools path: {args.tools_dir}")
    # 创建strace wrapper
    strace_path = args.tools_dir + '/' + "strace"
    wrapperStrace = WrapperStrace(pid, strace_path=strace_path, timeout=args.timeout)
    # 运行strace并获取结果
    res = wrapperStrace.run()

    # 准备输出
    output = sys.stderr
    if args.output:
        try:
            output = open(args.output, 'w')
        except IOError as e:
            sys.stderr.write(f"Error opening output file: {e}\n")
            sys.exit(1)
    
    mutex_addrs = []
    tid_waits = []
    for tid, mutex, status in res:
        mutex_addrs.append(mutex)
        tid_waits.append(tid)
    
    gdbserver_path = args.tools_dir + '/' + "gdbserver"
    # 初始化单个GDB会话
    gdb = GDBController(pid, host=args.host, port=args.port, symbol_file=args.symbol_file, gdbserver_path=gdbserver_path)
    try:
        for mutex in mutex_addrs:
            tid = gdb.get_owner(mutex)
            if tid is not None and int(tid) > 0:
                res.add((tid, mutex, "held"))
        stack_info = gdb.get_thread_stack_frame(tid_waits)
    finally:
        gdb.close()

    for tid, mutex, status in res:
        output.write(f"{tid}\t{mutex}\t{status}\n")
    if args.output:
        output.close()

    # 检测死锁环
    dlc = DeadLockChecker(list(res), stack_info)
    dlc.run()

if __name__ == '__main__':
    main()