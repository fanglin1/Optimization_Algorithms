# -*- coding = utf-8 -*-
# @Time : 2023/3/3 14:17
# @Author : zackf
# @File : GA_单代号网络.py
# @Software : PyCharm

import copy
import random
import math
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
        读取数据
        :param path: 数据文件路径
        :return: 结点数量，活动数量，结点对象，资源，活动信息，活动头尾结点
        '''
    data_file = open(path)
    data_content = data_file.read().split("\n")
    data_file.close()
    activity_num = int(data_content[0].split()[0])
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



# 生成初始可行解序列
def initialize(group_num, activity_num_, activity_node_list1):
    group_list = []
    for m in range(group_num):
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
        group_list.append(finished_activities)
            # print("本次选择：", activity_selected, "待选活动：", activities_waited, "活动数量", activity_num_, "活动总数量", len(finished_activities))
    return group_list


# 判断资源是否足够
def resource_check(resources_available, resources_need):
    '''
        检查资源是否足够，当剩余资源不足以支持被选活动消耗时，释放正在排班的活动
        :param activity: 被选择的活动
        :return:
        '''
    is_available = True
    for i in range(len(resources_available)):
        if resources_available[i] < resources_need[i]:
            is_available = False
    return is_available


def time_check(activity, node_list, res1):
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


def calculate_time(lists_, resources_, node_list1):
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
        spend_time_list.append(max(time_list))
    best_time = min(spend_time_list)
    best_list = lists_[spend_time_list.index(best_time)]
    return spend_time_list, best_time, best_list


def cross_over(parent_lists, group_size_):
    '''
    交叉操作
    :param parent_lists: 父代个体
    :return: 交叉形成的子代个体
    '''
    children_list = []
    children_num = group_size_-len(parent_lists)
    for i in range(int(children_num/2)):
        # 父代索引
        mother_index = random.randint(0, len(parent_lists) - 2)
        father_index = random.randint(mother_index + 1, len(parent_lists) - 1)
        # 父代个体
        mother_gene = parent_lists[mother_index]
        father_gene = parent_lists[father_index]
        # 交叉位置1和位置2
        gene_length = len(mother_gene)
        position1 = random.randint(0, gene_length-2)
        position2 = random.randint(position1+1, gene_length-1)
        # 生成子代1和子代2
        child1 = get_child(mother_gene, father_gene, position1, position2)
        child2 = get_child(father_gene, mother_gene, position1, position2)
        children_list.append(child1)
        children_list.append(child2)
    return children_list


def get_child(gene_1, gene_2, position1, position2):
    '''
    :param gene1: 父代个体1
    :param gene2: 父代个体1
    :param position1: 交叉位置1
    :param position2: 交叉位置2
    :return: 两个父代经交叉生成的子代
    '''
    gene1 = copy.deepcopy(gene_1)
    gene2 = copy.deepcopy(gene_2)
    list_ = []
    cross_num = position2-position1
    # 片段1
    for i in range(0, position1+1):
        value = gene1[0]
        # print(value)
        list_.append(value)
        gene1.remove(value)
        gene2.remove(value)
    # 片段2
    for i in range(0, cross_num):
        value = gene2[0]

        list_.append(value)
        gene1.remove(value)
        gene2.remove(value)
    # 片段3
    list_ += gene1
    return list_


def mutation(list1, node_list1):
    '''
    变异
    :param lists: 经交叉还未进行变异的子代种群
    :return: 经变异的子代种群
    '''
    mutated_list = []
    lists = copy.deepcopy(list1)
    node_list = copy.deepcopy(node_list1)
    for list_ in lists:
        # 判断该子代个体是否进行变异
        rand_num = random.random()
        if rand_num <= mutation_rate:

            for i in range(1, len(list_)-1):
                # 判断该位置是否与前面位置交换
                exchange_rate = 0.3
                rand_num2 = random.random()
                if rand_num2 <= exchange_rate:
                    activity = list_[i]
                    pre_activities = node_list[activity].predecessors
                    # 位置交换
                    if list_[i-1] not in pre_activities:
                        list_[i], list_[i-1] = list_[i-1], list_[i]
            mutated_list.append(list_)
        else:
            mutated_list.append(list_)
    return mutated_list


def select(lists, resource_, node_list1):
    '''
    锦标赛算法选择
    :param lists: 种群列表，每个元素代表种群内一个个体
    :return: 选择后产生的子代
    '''

    spend_time, best, best_list = calculate_time(lists, resource_, node_list1)

    group_min_time = min(spend_time)
    index = spend_time.index(group_min_time)
    best_list = lists[index]


    # 计算适应度
    fit_value = []  # 存储每个个体的适应度
    for i in range(len(spend_time)):
        fit_value.append(1 / spend_time[i] ** 2)
    parent_num = len(lists)
    # 适应度总和
    total_fit = 0
    for i in range(len(fit_value)):
        total_fit += fit_value[i]

    # 计算累计概率
    t = 0
    for i in range(len(fit_value)):
        t = t + fit_value[i]
        fit_value[i] = t

    # 生成随机数序列用于选择和比较
    ms = []  # 随机数序列
    for i in range(len(lists)):
        ms.append(random.uniform(0,total_fit))
    ms.sort()

    # 轮盘赌选择法
    i = 0
    j = 0
    parents = []
    while i < len(lists):
        # 选择--累积概率大于随机概率
        if (ms[i] < fit_value[j]):
            if lists[j] not in parents:
                parents.append(lists[j])
            i = i + 1
        # 不选择--累积概率小于随机概率
        else:
            j = j + 1

    # winner_num = 5
    # select_iter_num = math.floor(group_size/winner_num)
    # for i in range(select_iter_num):
    #     winner_list = []
    #     competitors = random.sample(list(range(parent_num)), 10)
    #     competitor_time_list = []
    #     for j in competitors:
    #         competitor_time_list.append(spend_time[j])
    #     while len(winner_list) < winner_num:
    #         winner_time = min(competitor_time_list)
    #         winner_competitor_index = competitor_time_list.index(winner_time)
    #         winner_index = competitors[winner_competitor_index]
    #         winner_list.append(winner_index)
    #         children.append(lists[winner_index])
    #         competitor_time_list.remove(winner_time)
    # if len(children) < group_size:
    #     num = len(lists)-len(children)
    #     t = copy.deepcopy(spend_time)
    #     for i in range(num):
    #         time_ = min(t)
    #         index = t.index(time_)
    #         t[index] = 0
    #         children.append(lists[index])
    return parents, best, best_list


if __name__ == "__main__":
    file_path = "RCPSP_J601_GA.xlsx"
    result = []

    start_time = time.time()
    # 交叉比例
    cross_rate = 0.5
    # 变异概率
    mutation_rate = 0.2
    # 跌代次数
    iter_num = 2000
    # 获取数据
    activity_num, activity_node_list, resource_num, resources = get_data("J601_1.RCP")
    group_size = (activity_num-1)*activity_num
    # 初始化种群
    old_group = initialize(group_size, activity_num, activity_node_list)
    best_time = sys.maxsize
    best_individual = []
    for i in range(iter_num):
        # 交叉
        children_group = cross_over(old_group, group_size)
        # print("交叉完成")
        # 变异
        mutated_children = mutation(children_group, activity_node_list)
        # print("变异完成")
        total_group = old_group + mutated_children
        # 选择
        old_group, best, best_list = select(total_group, resources, activity_node_list)
        # print("选择完成")
        if best_time == 0 or best < best_time:
            best_time = best
            best_individual = copy.copy(best_list)
        end_time = time.time()
        result.append([0, i, end_time - start_time, best_time])
    df = pd.DataFrame(result, columns=None)
    df.to_excel(file_path)
