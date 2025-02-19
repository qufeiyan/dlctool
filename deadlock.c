#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

// 定义两个互斥锁
pthread_mutex_t mutex1, mutex2;

// 线程 1 函数
void* thread1_function(void* arg) {
    // 线程 1 先锁定 mutex1
    pthread_mutex_lock(&mutex1);
    printf("线程 1 已锁定 mutex1\n");

    // 模拟一些工作
    sleep(1);

    // 线程 1 尝试锁定 mutex2
    printf("线程 1 正在尝试锁定 mutex2\n");
    pthread_mutex_lock(&mutex2);
    printf("线程 1 已锁定 mutex2\n");

    // 解锁 mutex2
    pthread_mutex_unlock(&mutex2);
    // 解锁 mutex1
    pthread_mutex_unlock(&mutex1);

    return NULL;
}

// 线程 2 函数
void* thread2_function(void* arg) {
    // 线程 2 先锁定 mutex2
    pthread_mutex_lock(&mutex2);
    printf("线程 2 已锁定 mutex2\n");

    // 模拟一些工作
    sleep(1);

    // 线程 2 尝试锁定 mutex1
    printf("线程 2 正在尝试锁定 mutex1\n");
    pthread_mutex_lock(&mutex1);
    printf("线程 2 已锁定 mutex1\n");

    // 解锁 mutex1
    pthread_mutex_unlock(&mutex1);
    // 解锁 mutex2
    pthread_mutex_unlock(&mutex2);

    return NULL;
}

int main() {
    pthread_t thread1, thread2;

    // 初始化两个互斥锁
    pthread_mutex_init(&mutex1, NULL);
    pthread_mutex_init(&mutex2, NULL);

    // 创建线程 1
    if (pthread_create(&thread1, NULL, thread1_function, NULL) != 0) {
        perror("线程 1 创建失败");
        return 1;
    }

    // 创建线程 2
    if (pthread_create(&thread2, NULL, thread2_function, NULL) != 0) {
        perror("线程 2 创建失败");
        return 1;
    }

    // 等待线程 1 结束
    if (pthread_join(thread1, NULL) != 0) {
        perror("等待线程 1 结束失败");
        return 1;
    }

    // 等待线程 2 结束
    if (pthread_join(thread2, NULL) != 0) {
        perror("等待线程 2 结束失败");
        return 1;
    }

    // 销毁两个互斥锁
    pthread_mutex_destroy(&mutex1);
    pthread_mutex_destroy(&mutex2);

    return 0;
}