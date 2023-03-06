# -*- coding = utf-8 -*-
# @Time : 2023/2/27 19:01
# @Author : zackf
# @File : PSO_单代号网络.py
# @Software : PyCharm
import copy
import random
import sys
import time
import pandas as pd

class ActivityNode:
    def __init__(self, its_id, duration, resource, successors):
        self.its_id = its_id
        self.duration = duration
        self.resource = resource
        self.successors = successors
        self.predecessors = []


def get_data(path):
    '''
    获取数据
    :param path: 文件路径
    :return: 活动数量，网络图，资源种类，资源数量
    '''
    data_file = open(path)
    data_content = data_file.read().split("\n")
    data_file.close()
    activity_num = int(data_content[0].split()[0])
    resource_num = int(data_content[0].split()[1])
    resources = [int(x) for x in data_content[1].split()]
    # print(activity_num)
    # print(resource_num)
    # print(resources)
    activity_node_list = []
    activity_pre_list = [[] for x in range(activity_num)]
    for content in data_content[2:]:
        info = content.split()
        current_id = len(activity_node_list)
        duration_ = int(info[0])
        resources_need = []
        successors_ = []
        for i in range(1, 1+resource_num):
            resources_need.append(int(info[i]))
        for i in range(2+resource_num, len(info)):
            suc_activity_id = int(info[i])-1
            successors_.append(suc_activity_id)
            activity_pre_list[suc_activity_id].append(current_id)
        activity_node = ActivityNode(current_id, duration_, resources_need, successors_)

        activity_node_list.append(activity_node)
    # 前序活动
    for i in range(len(activity_node_list)):
        activity_node_list[i].predecessors = activity_pre_list[i]


    return activity_num, activity_node_list, resource_num, resources


# 随机为每个粒子生成一个优先级序列和速度序列
def initialize_priority_and_velocity(group_size, activity_num, max_position, min_position, max_v, min_v):
    '''
    初始化各粒子各个活动的优先值和速度
    :param group_size: 种群大小
    :param activity_num: 活动数量
    :param max_position: 最大优先值
    :param min_position: 最小优先值
    :param max_v: 最大速度
    :param min_v: 最小速度
    :return:
    '''
    priority_list = []
    velocity_list = []
    for i in range(group_size):
        priority = [random.uniform(min_position, max_position) for i in range(0, activity_num)]
        priority_list.append(priority)
        velocity = [random.uniform(min_v, max_v) for i in range(0, activity_num)]
        velocity_list.append(velocity)
    return priority_list, velocity_list


# 根据优先度列表选生成活动列表
def get_schedule_through_priority(priority_list, activity_node_list1):
    '''

    :param priority_list: 各粒子的活动的优先值列表
    :param activity_node_list1: 网络计划图
    :return: 活动安排序列
    '''
    schedule_list = []
    activity_num_ = len(priority_list[0])
    activity_node_list1_ = activity_node_list1
    # count = 0
    for priority in priority_list:
        # print("priority\n", priority)
        # print(activity_node_list2)
        finished_activities = [0]
        activity_node_list2 = copy.deepcopy(activity_node_list1_)
        activities_waited = activity_node_list2[finished_activities[0]].successors
        if len(activities_waited) == 0:
            print("未访问到")
        # print("初始化：", activity_num_, finished_activities, activity_node_list2[finished_activities[0]].successors)
        while len(finished_activities) < activity_num_:
            # 选择一个活动执行,此处不再是随机选择一个活动，而是在等待选择的活动中选择优先度高的活动
            # print(len(finished_activities), activity_num_)
            # activity_selected = random.choice(activities_waited)
            # 获取等待选择的活动的优先度
            # print("activities_waited", activities_waited)
            priority_activities_waited = [0 for _ in activities_waited]
            for i in range(len(activities_waited)):
                priority_activities_waited[i] = priority[activities_waited[i]]
            max_priority = max(priority_activities_waited)
            max_priority_index = priority_activities_waited.index(max_priority)
            activity_selected = activities_waited[max_priority_index]
            finished_activities.append(activity_selected)
            activities_waited.remove(activity_selected)
            # 更新已完成和待完成活动列表
            if len(finished_activities) != activity_num_:
                next_activities = activity_node_list2[activity_selected].successors
                for activity in next_activities:
                    pred_activities = activity_node_list2[activity].predecessors
                    have_finished = True
                    for pre_activity in pred_activities:
                        if pre_activity not in finished_activities:
                            have_finished = False
                            break
                    if have_finished:
                        activities_waited.append(activity)
        schedule_list.append(finished_activities)
        # print("完成了第",i,"次")
    return schedule_list


# 判断资源是否足够
def resource_check(resources_available, resources_need):
    '''
    资源约束
    :param resources_available: 可用资源
    :param resources_need: 需要资源
    :return:
    '''
    is_available = True
    for i in range(len(resources_available)):
        if resources_available[i] < resources_need[i]:
            is_available = False
    return is_available


def time_check(activity, node_list, res1):
    '''
    时序约束，检查活动的前序活动是否都已完成
    :param activity: 被选择的活动
    :param node_list: 网络图
    :param res1: 正在进行的活动
    :return:
    '''
    is_finished = True
    pre_activity_max_time = 0
    if len(res1):
        res_activity = [t[0] if t[0] != activity else t[0] for t in res1]
        # print(res_activity)
        for m in node_list[activity].predecessors:
            if m in res_activity:
                is_finished = False
                res_index = res_activity.index(m)
                time = res1[res_index][1]
                if time > pre_activity_max_time:
                    pre_activity_max_time = time
    return is_finished, pre_activity_max_time


# 根据生成的活动列表计算时间
def calculate_time(lists_, resources_, node_list1):
    '''
    计算序列适应度（时间）
    :param lists: 当前解
    :param resources_:资源总量
    :param node_list1:活动网络
    :return: 时间
    '''
    spend_time_list = []
    node_list = copy.deepcopy(node_list1)
    for individual in lists_:
        current_time = 0
        time_list = []
        res_time = []
        resource_list = copy.deepcopy(resources_)
        for i in individual:
            # 前序活动检查，活动开始时间需要至少其前序活动都已完成
            pre_activity_check, max_pre_time = time_check(i, node_list, res_time)
            # 如果前序活动还没有全部完成，则需要释放前序活动及未完成的活动
            if not pre_activity_check:
                # 释放前序活动
                current_time = max_pre_time

                for res in res_time:
                    # print("res", res_time)
                    if res[1] <= max_pre_time:
                        # print("最大时间", max_pre_time, "活动结束时间", res[1])
                        resource_give_off = node_list[res[0]].resource
                        for j in range(len(resource_list)):
                            resource_list[j] += resource_give_off[j]
                        # print("根据前序释放了", res[0], node_list[res[0]].resource, "剩余", resource_list)

                res_time = [res1 for res1 in res_time if res1[1] > max_pre_time]
            while True:
                # 资源约束检查，如果资源不够则需要释放资源
                resources_need = node_list[i].resource
                resource_enough = resource_check(resource_list, resources_need)
                if resource_enough:
                    for j in range(len(resource_list)):
                        resource_list[j] -= resources_need[j]
                    res_time.append([i, current_time + node_list[i].duration])
                    time_list.append(current_time + node_list[i].duration)
                    # print([i, current_time+node_list[i].duration, "消耗", resources_need,"剩余", resource_list])
                    # print("时间", current_time+node_list[i].duration)
                    break
                elif res_time:
                    min_time = min(t[1] for t in res_time)
                    current_time = min_time
                    for res in res_time:
                        if res[1] <= min_time:
                            resource_give_off = node_list[res[0]].resource
                            for j in range(len(resource_list)):
                                resource_list[j] += resource_give_off[j]
                            # print("释放了", res[0], node_list[res[0]].resource, "剩余", resource_list)
                    res_time = [res1 for res1 in res_time if res1[1] > min_time]
                    # res_time.remove(res)
                else:
                    print("出现错误,活动编号：", i, "所需资源：", resources_need, "剩余资源", resource_list)
                    break
        spend_time_list.append(max(time_list))
    best_time = min(spend_time_list)
    index = spend_time_list.index(best_time)
    best_list = lists_[spend_time_list.index(best_time)]
    return spend_time_list, best_time, best_list, index


# PSO主体
if __name__ == "__main__":
    file_path = "RCPSP_J601_PSO.xlsx"
    result = []
    # for t in range(5):
    #     print(t)
    start_time = time.time()
    # random.seed(6)
    x_min = 0
    x_max = 10
    v_max = 2
    v_min = -v_max
    w = 0.81
    c1 = c2 = 0.07

    iter_num = 2000
    activity_num_, activity_node_list_, resource_num_, resources_list = get_data('J601_1.RCP')
    group_num = (activity_num_-1)*activity_num_
    # 初始化优先度值和速度值
    priority_list1, velocity_list1 = initialize_priority_and_velocity(group_num, activity_num_, x_max, x_min, v_max, v_min)
    # 根据各粒子的活动的优先度值产生活动虚了
    schedule_list1 = get_schedule_through_priority(priority_list1, activity_node_list_)
    # 计算活动序列花费的世界
    old_best_time_list, current_best_time, current_best_seq, best_index = calculate_time(schedule_list1, resources_list, activity_node_list_)
    # 初始化个体最优位置和全局最优位置
    global_best_time = current_best_time
    global_best_seq = current_best_seq
    global_best_priority = priority_list1[best_index]
    self_best_position_list = priority_list1

    for it in range(iter_num):
        # 更新速度列表
        for row_index in range(len(velocity_list1)):
            for col_index in range(activity_num_):
                # 自己当前速度在这个维度的值
                self_velocity = velocity_list1[row_index][col_index]
                # 自己当前位置在这个维度的值
                self_position = priority_list1[row_index][col_index]
                # 群体内最好的个体的这个维度的值
                best_position = global_best_priority[col_index]
                # best_velocity_ = best_velocity[col_index]
                # 自己走过的最好的位置在这个维度的值
                self_best_position = self_best_position_list[row_index][col_index]
                rand1 = random.random()
                rand2 = random.random()
                value = w*self_velocity + c1*rand1*(self_best_position-self_position) + c2*rand2*(best_position-self_position)
                # print(self_velocity, self_position, best_position, self_best_position)
                if value > v_max:
                    value = v_max
                if value < v_min:
                    value = v_min
                velocity_list1[row_index][col_index] = value
        # 更新优先值列表
        for row_index in range(len(priority_list1)):
            for col_index in range(activity_num_):
                # 自己当前位置在这个维度的值
                self_position = priority_list1[row_index][col_index]
                value = velocity_list1[row_index][col_index] + self_position
                if value > x_max:
                    value = x_max
                if value < x_min:
                    value = x_min
                priority_list1[row_index][col_index] = value
        # 重新根据更新的优先度列表来计算工序列表
        schedule_list1 = get_schedule_through_priority(priority_list1, activity_node_list_)
        # 重新计算工序时间
        new_best_time_list, current_best_time, current_best_seq, best_index = calculate_time(schedule_list1, resources_list, activity_node_list_)
        # 更新每个粒子的最好位置
        for m in range(len(new_best_time_list)):
            if old_best_time_list[m] > new_best_time_list[m]:
                old_best_time_list[m] = new_best_time_list[m]
                self_best_position_list[m] = priority_list1[m]
        # 更新全局最优粒子及全局最优完成时间
        if current_best_time < global_best_time:
            global_best_time = current_best_time
            global_best_seq = current_best_seq
            global_best_priority = priority_list1[best_index]
        end_time = time.time()
        result.append([0, it, end_time - start_time, global_best_time])
    df = pd.DataFrame(result, columns=None)
    df.to_excel(file_path)











