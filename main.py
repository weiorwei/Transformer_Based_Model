# # encoding: utf-8
# a = input("please input a number:")
# print("hello world")
#
#
# class S1():
#     def find_len(self,nums, x, k):
#         length=len(nums)
#         min_index=[]
#         if x<len(nums):
#             for i in range(2):
#                 if i == 0:
#                     low = 0
#                     high=x
#                 else:
#                     low = x
#                     high = length-1
#                 while low<=high:
#                     mid=(low+high)//2
#                     if nums[mid]==k:
#                         min_index.append(abs(mid-x))
#                         if i ==0:
#                             low=mid+1
#                         else:
#                             high=mid-1
#                     elif k !=nums[mid]:
#                         if i ==0:
#                             low=mid+1
#                         else:
#                             high=mid-1
#         else:
#                 low = 0
#                 high=length-1
#                 while low<=high:
#                     mid=(low+high)//2
#                     if nums[mid]==k:
#                         min_index.append(abs(mid-x))
#                         low=mid+1
#                     elif k !=nums[mid]:
#                         low=mid+1
#         if min_index:
#             return min(min_index)
#         else:
#             return -1
# s1=S1()
# print(s1.find_len([1,2,3,1,2,3,1,2,3,3,2,3],5,1))
# print(s1.find_len([1, 1, 2 ,1 ,3, 2 ,2 ,3 ,3],1,3))
#
# class S1():
#     def find_len(self,nums, x, k):
#         if x<len(nums):
#             min_index=[]
#             for i in range(x):
#                 if nums[x-i]==k:
#                     min_index.append(i)
#                     break
#             for i in range(len(nums)-x-1):
# #                 if nums[x+i]==k:
# #                     min_index.append(i)
# #                     break
# #             return min(min_index)
# #         else:
# #             min_index = []
# #             for i in range(len(nums)):
# #                 if nums[len(nums) - i] == k:
# #                     min_index.append(i)
# #                     break
# #             return min(min_index)
# #
# # s1=S1()
# # print(s1.find_len([1,2,3,1,2,3,1,2,3,3,2,3],5,1))
# # print(s1.find_len([1, 1, 2 ,1 ,3, 2 ,2 ,3 ,3],1,3))
# #
# # class S2():
# #     def find_len(self,nums):
# #         times=0
# #         max_search=len(str(nums))*9
# #         for i in range(max_search):
# #             sum_nums = nums-i
# #             for k in range(len(str((nums-i)))):
# #                 sum_nums=sum_nums+int(str(nums-i)[k])
# #             if sum_nums == nums:
# #                 times=times+1
# #         return times
# #
# #
# # s2=S2()
# # print(s2.find_len(21))
#
# # def fun(li):
# # 	return li[1]
# # random = [(2, 2), (3, 4), (4, 1), (1, 3)]
# # list_n=[1,2,4,6,2,1,9,3,5,6]
# # random.sort(key=lambda x: x[1])
# # print(random)
#
#
# class S():
#     def switch(self,arry,low,high):
#         standard=arry[low]
#         while low<high:
#             while low<high and arry[high]>standard:
#                 high=high-1
#             arry[low]=arry[high]
#             while low<high and arry[low]<standard:
#                 low=low+1
#             arry[high]=arry[low]
#
#         arry[low]=standard
#
#         return low,arry
#
#     def quick_line(self,nums,low,high):
#         if low>=high:
#             return nums
#
#         mid,nums=self.switch(nums,low,high)
#         nums=self.quick_line(nums, low, mid-1)
#         nums=self.quick_line(nums, mid + 1, high)
#
#         return nums
#
# # list_ns=[1,3,5,2,4,8,9,2,1,3,4,2,1,5,6,3,2]
# list_ns=[4,3,5,2,1]
# s=S()
# print(s.quick_line(list_ns,0,len(list_ns)-1))
#
#
# def quick_sort(alist, start, end):
#     """快速排序"""
#     if start >= end:  # 递归的退出条件
#         return
#     mid = alist[start]  # 设定起始的基准元素
#     low = start  # low为序列左边在开始位置的由左向右移动的游标
#     high = end  # high为序列右边末尾位置的由右向左移动的游标
#     while low < high:
#         # 如果low与high未重合，high(右边)指向的元素大于等于基准元素，则high向左移动
#         while low < high and alist[high] >= mid:
#             high -= 1
#         alist[low] = alist[high]  # 走到此位置时high指向一个比基准元素小的元素,将high指向的元素放到low的位置上,此时high指向的位置空着,接下来移动low找到符合条件的元素放在此处
#         # 如果low与high未重合，low指向的元素比基准元素小，则low向右移动
#         while low < high and alist[low] < mid:
#             low += 1
#         alist[high] = alist[low]  # 此时low指向一个比基准元素大的元素,将low指向的元素放到high空着的位置上,此时low指向的位置空着,之后进行下一次循环,将high找到符合条件的元素填到此处
#
#     # 退出循环后，low与high重合，此时所指位置为基准元素的正确位置,左边的元素都比基准元素小,右边的元素都比基准元素大
#     alist[low] = mid  # 将基准元素放到该位置,
#     # 对基准元素左边的子序列进行快速排序
#     quick_sort(alist, start, low - 1)  # start :0  low -1 原基准元素靠左边一位
#     # 对基准元素右边的子序列进行快速排序
#     quick_sort(alist, low + 1, end)
#     return alist
#
# print(quick_sort(list_ns,0,len(list_ns)-1))


# import torch.nn as nn
# import torch
#
# class model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.con1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3))
#         self.con2=nn.Conv2d(in_channels=16,out_channels=64,kernel_size=(3,3))
#
#         self.fullyconnet=nn.Linear(in_features=64*60*60,out_features=2,bias=True)
#
#     def forward(self,x):
#         x=self.con2(self.con1(x))
#         x=nn.Flatten()(x)
#         x=self.fullyconnet(x)
#         x=nn.Softmax()(x)
#         return x
#
# input_value=torch.randn((1,3,64,64))
# model=model()
# out=model(input_value)
# def sort_nums(nums):
#     for i in range(len(nums)):
#         for k in range(i):
#             if nums[k]>nums[k+1]:
#                 mid=nums[k]
#                 nums[k]=nums[k+1]
#                 nums[k+1]=mid
#     return nums
#
#
# nums=[1,3,2,5,6]
# print(sort_nums(nums))
#
# value=[]
#
# list_add=[]
#
# v=value[0]
# for
#     if value[i]>v
#         v=value[i]
#         m+=1
#     else:






#
# import numpy as np
# array=np.arange(0,11,(3,2))








#
# class S():
#     def __init__(self,na,s):
#         self.na=na
#         self.s=s
#
# b=S("a","b")
# a=S("c","d")
#
# b.h="a"
# a.s=("10")
# print(a.h)
