# -*- coding = utf-8 -*-
# @Time : 2023/3/3 13:57
# @Author : zackf
# @File : SA_单代号网络.py
# @Software : PyCharm

import copy
import random
import math
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
    # 资源总量
    resource_num = int(data_content[0].split()[1])
    resources = [int(x) for x in data_content[1].split()]

    activity_node_list = []
    activity_pre_list = [[] for x in range(activity_num)]
    for content in data_content[2:]:
        info = content.split()
        current_id = len(activity_node_list)
        duration_ = int(info[0])
        resources_need = []
        successors_ = []
        # 资源消耗
        for i in range(1, 1+resource_num):
            resources_need.append(int(info[i]))
        # 后序
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


# 生成初始可行解序列
def initialize(activity_num_, activity_node_list1):
    '''
    初始化当前解
    :param activity_num_: 活动数
    :param activity_node_list1: 活动网络图
    :return: 符合前后序约束的活动序列
    '''
    finished_activities = [0]
    activity_node_list_ = copy.deepcopy(activity_node_list1)
    activities_waited = activity_node_list_[finished_activities[0]].successors
    # print("初始化：", activity_num_, finished_activities, activity_node_list_[finished_activities[0]].successors)
    while len(finished_activities) < activity_num_:
        # 选择一个活动执行
        # print(len(finished_activities), activity_num_)
        activity_selected = random.choice(activities_waited)
        finished_activities.append(activity_selected)
        activities_waited.remove(activity_selected)
        # 更新已完成和待完成活动列表
        if len(finished_activities) != activity_num_:
            next_activities = activity_node_list_[activity_selected].successors
            for activity in next_activities:
                pred_activities = activity_node_list_[activity].predecessors
                have_finished = True
                for pre_activity in pred_activities:
                    if pre_activity not in finished_activities:
                        have_finished = False
                        break
                if have_finished:
                    activities_waited.append(activity)
        # print("本次选择：", activity_selected, "待选活动：", activities_waited, "活动数量", activity_num_, "活动总数量", len(finished_activities))
    return finished_activities


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
        # 正在进行的活动编号（需要排除activity）
        res_activity = [t[0] if t[0] != activity else t[0] for t in res1]
        # print(res_activity)
        # activity的前序活动检查
        for m in node_list[activity].predecessors:
            # 如果前序活动还未结束，则记录最晚结束的活动的时间
            if m in res_activity:
                is_finished = False
                res_index = res_activity.index(m)
                time = res1[res_index][1]
                if time > pre_activity_max_time:
                    pre_activity_max_time = time
    return is_finished, pre_activity_max_time


def calculate_time(lists, resources_, node_list1):
    '''
    计算序列适应度（时间）
    :param lists: 当前解
    :param resources_:资源总量
    :param node_list1:活动网络
    :return: 时间
    '''
    current_time = 0
    time_list = []
    res_time = []
    resource_list = copy.deepcopy(resources_)
    node_list = copy.deepcopy(node_list1)
    for i in lists:
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
                res_time.append([i, current_time+node_list[i].duration])
                time_list.append( current_time+node_list[i].duration)
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
                print("出现错误,活动编号：",i, "所需资源：", resources_need, "剩余资源", resource_list)
                break
    spend_time=max(time_list)
    return spend_time



# 产生邻域解
def generate_new_list(old_list,node_list1):
    '''
    产生邻域解，随机选择一个位置a，查找这个位置的活动最后一个前序活动p和最早的后续活动的位置s，
    在这p/s两个位置之间随机选择一个位置b与a进行交换
    :param old_list:旧的序列解
    :return:新序列解
    '''
    new_list = []
    node_list = copy.deepcopy(node_list1)
    while True:
        list_content = old_list
        change_position = random.randint(1, len(list_content) - 1)
        change_element = list_content[change_position]
        # 初始化最后一个前序活动和第一个后续活动
        last_pre = first_suc = change_element
        # 找最后一个前序活动
        for j in list_content[0: change_position]:
            if j in node_list[change_element].predecessors:
                last_pre = j
        # 找第一个后序活动
        for j in list_content[change_position:]:
            if j in node_list[change_element].successors:
                first_suc = j
                break
        # 最后一个前序活动和第一个后续活动的位置
        last_pre_position = list_content.index(last_pre)
        first_suc_position = list_content.index(first_suc)
        # 判断是否可以交换
        if (change_position - last_pre_position) <= 1 and (first_suc_position - change_position) <= 1:
            continue
        else:
            # 如果可以交换，则选择交换位置
            position_list = list(range(last_pre_position + 1, first_suc_position))
            # print("被选择的位置", change_position, "最后一个前继活动位置", last_pre_position, "第一个后继活动位置", first_suc_position)
            if change_position >= position_list[0]:
                # print("移除了")
                position_list.remove(change_position)

            change_position_2 = random.choice(position_list)
            list_content.remove(change_element)
            # print("位置1:", change_position, "位置2：", change_position_2)
            list_content.insert(change_position_2, change_element)
            new_list = list_content
            break
    if len(new_list):
        return new_list
    else:
        print("新序列生成有误")


if __name__ == "__main__":
    file_path = "RCPSP_J601_SA.xlsx"
    result = []

    start_time = time.time()
    # 读取数据
    activity_num, activity_node_list, resource_num, resources = get_data("J601_1.RCP")
    old_gen = initialize(activity_num, activity_node_list)
    old_time = calculate_time(old_gen, resources, activity_node_list)
    min_time = old_time
    min_seq = old_gen
    T0 = 1000  # 初始温度
    T = T0  # 第一次迭代温度
    maxGen = 2000  # 最大迭代次数
    lk = (activity_num-1)*activity_num # 每个温度下的迭代次数
    al = 0.97  # 温度衰减系数
    for outer_iter in range(maxGen):
        for inner_iter in range(lk):
            # 产生邻域解
            new_list = generate_new_list(old_gen, activity_node_list)
            # 适应度（时间）计算
            time_spend = calculate_time(new_list, resources, activity_node_list)
            if time_spend < old_time:
                old_gen = new_list
                old_time = time_spend
            else:
                # 适应度小于旧解时，概率接受新解
                p = math.exp(-(time_spend-old_time)/T)
                r = random.random()
                if p>r:
                    old_gen = new_list
                    old_time = time_spend
            # 更新最优解
            if min_time > old_time:
                min_time = old_time
                min_seq = copy.copy(old_gen)
            # print(min_time)
        # 降温
        T = al*T
        end_time = time.time()
        result.append([0, outer_iter, end_time - start_time, min_time])
    df = pd.DataFrame(result, columns=None)
    df.to_excel(file_path)
