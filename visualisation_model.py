# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
from pathlib import Path

dataset = pd.read_csv('P:/Cloud_counselage/DS_DATESET.csv')
dataset = dataset.drop(['Certifications/Achievement/ Research papers','Link to updated Resume (Google/ One Drive link preferred)','link to Linkedin profile'], axis = 1)
dataset.dropna().head(1)


with PdfPages('data_visualization.pdf') as pdf:
    
    
    #A. The number of students applied to different technologies.
    
    areas = np.array(['Artificial Intelligence','Big Data','Cloud Computing ','IoT ','Digital Marketing ','Python ','QMS/Testing ','Data Science ','Machine Learning','Blockchain ','RPA','DevOps ','Web Development ','Cyber Security ', 'Mobility','Information Security'])
    areas.sort()
    i = dataset.groupby('Areas of interest')['First Name'].count()
    
    plt.figure(1)
    plt.figure(figsize=(10,8))
    explode = (0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2)
    def make_autopct(i):
        def my_autopct(pct):
            total = sum(i)
            val = int(round(pct*total/100.0))
            return '{v:d}'.format(v=val)
        return my_autopct
    
        
    plt.title("Students Applying For Different Technologies\n\n", fontsize=20)
    plt.pie(i,labels = areas,explode = explode,shadow = True,radius = 1,autopct = make_autopct(i))
    pdf.savefig()  
    plt.close()
    
    
    
    #B.  The number of students applied for Data Science who knew ‘’Python” and who didn’t.
    
    df1 = dataset.filter(['Areas of interest','Programming Language Known other than Java (one major)'], axis=1)
    df1 = df1.loc[df1['Areas of interest']== 'Data Science ']
    df1.rename(columns = {'Programming Language Known other than Java (one major)':'Knew Python'}, inplace = True)
    df1['Knew Python'] = np.where(df1['Knew Python'] == 'Python', 'Yes','No')
    df1.reset_index()
    plt.figure(2)
    b = plt.pie(x=df1['Knew Python'].value_counts(), labels=["Don't know python", "Know Python"],autopct='%1.1f%%')
    plt.title("Students have applied for data science", fontsize=20)
    pdf.savefig()  
    plt.close()
    
    
    #C.  The different ways students learned about this program.
    
    modes = ['Blog post','Ex/Current Employee','Facebook','Friend','Intern','LinkedIn','Newspaper','Other','Twitter']
    dw_count = dataset.groupby('How Did You Hear About This Internship?')['First Name'].count()
    
    plt.figure(3)
    plt.figure(figsize=(8,8))
    explode = (0.3,0.2,0.3,0.2,0.3,0.2,0.3,0.2,0.3)
    def make_autopct(dw_count):
        def my_autopct(pct):
            total = sum(i)
            val = int(round(pct*total/100.0))
            return '{v:d}'.format(v=val)
        return my_autopct
    
        
    plt.title("The different ways student learned about this program\n\n",fontsize = 20)
    plt.pie(dw_count,labels = modes,explode = explode,shadow = True,radius = 1,autopct = make_autopct(i))
    pdf.savefig()  
    plt.close()
    
    #D.   Students who are in the fourth year and have a CGPA greater than 8.0. 
    
    df3 = dataset.filter(['Which-year are you studying in?','CGPA/ percentage'], axis=1)
    df3 = df3.loc[df3['Which-year are you studying in?']=='Fourth-year']
    students_above_8 = len([x for x in df3['CGPA/ percentage'] if x >= 8.0])
    total_CategoryWise = [students_above_8, len(df3)-students_above_8]
    plt.figure(4)
    d= plt.pie(total_CategoryWise, labels=["Having CGPA >= 8", "Having CGPA < 8"], autopct='%1.1f%%')
    plt.title("Students who are in Fourth-year", fontsize=20)
    pdf.savefig()  
    plt.close()
    
    #E.   Students who applied for Digital Marketing with verbal and written communication score greater than 8. 
    
    
    df4 = dataset.filter(['Areas of interest','Rate your written communication skills [1-10]','Rate your verbal communication skills [1-10]'])
    df4 = df4.loc[df4['Areas of interest']=='Digital Marketing ']
    df4.rename(columns = {'Rate your verbal communication skills [1-10]':'Verbal Score'}, inplace = True)
    df4.rename(columns = {'Rate your written communication skills [1-10]':'Written Score'}, inplace = True)
    score_above_8 = len([x for x in zip(df4['Verbal Score'], df4['Written Score']) if (x[0] >= 8 and x[1] >= 8)])
    tot_cat_wise = [score_above_8, len(df4)-score_above_8]
    plt.figure(5)
    e = plt.pie(tot_cat_wise, labels=["Written and Verbal Scores Above or Equal to 8", "Written and Verbal Scores Below 8"], autopct='%1.1f%%')
    plt.title("Students who applied for Digital Marketing", fontsize=20)
    pdf.savefig()  
    plt.close()
    
    
    #F.   Year-wise and area of study wise classification of students
    
    df5 = dataset.filter(['Which-year are you studying in?','Major/Area of Study'])
    df5.rename(columns = {'Which-year are you studying in?':'Study year'}, inplace = True)
    df5.rename(columns = {'Major/Area of Study':'Area of Study'}, inplace = True)
    plt.figure(6)
    plt.figure(figsize=(8,9))
    sns.set_style('darkgrid')
    f = sns.countplot(x='Study year',hue = 'Area of Study',data = df5, palette='YlOrRd')
    plt.xticks(rotation=90)
    plt.title(label='Classification on basis of year and area of study')
    pdf.savefig()  
    plt.close()
    
    #G.    City and college wise classification of students
    
    df6 = dataset.filter(['City','College name'])
    plt.figure(7)
    plt.figure(figsize=(8,9))
    sns.set_style('darkgrid')
    g = sns.countplot(x='City',data = df6, palette = 'Blues')
    plt.xticks(rotation=90)
    plt.title(label='Classification on basis of city')
    pdf.savefig()  
    plt.close()
    
    plt.figure(8)
    plt.figure(figsize=(8,12))
    sns.set_style('darkgrid')
    h = sns.countplot(x='College name',data = df6, palette = 'rainbow')
    plt.xticks(rotation=90)
    plt.title(label='Classification on basis of college')
    pdf.savefig()  
    plt.close()
    
    
     # H. Plot the relationship between the CGPA and the target variable
    data1 = dataset.filter(['CGPA/ percentage','Label'])
    eligible = data1[data1['Label']=='eligible']
    s1 = len([x for x in eligible['CGPA/ percentage'] if (x<8 and x>=7)])
    s2 = len([x for x in eligible['CGPA/ percentage'] if (x<9 and x>=8)])
    s3 = len([x for x in eligible['CGPA/ percentage'] if (x>=9)])

    s1t = len([x for x in data1['CGPA/ percentage'] if (x<8 and x>=7)])
    s2t = len([x for x in data1['CGPA/ percentage'] if (x<9 and x>=8)])
    s3t = len([x for x in data1['CGPA/ percentage'] if (x>=9)])

    df7 = pd.DataFrame({'eligible': [s1,s2,s3], 'ineligible': [s1t-s1,s2t-s2,s3t-s3]})
    plt.figure(9)
    plt.figure(figsize = (10,5))
    h = df7.plot(kind='bar')
    plt.title("Relationship between CGPA and eligibility", fontsize=20)
    h.set_ylabel("Number of Students")
    h.set_xlabel("Range of CGPA")
    h.set_xticklabels(["7 - 8", "8 - 9", "9 - 10"],rotation=0)
    pdf.savefig()  
    plt.close()
    
    # I. Plot the relationship between the Area of Interest and the target variable
    data2 = dataset.filter(['Areas of interest','Label'])
    eligible = data2[data2['Label'].str.contains('eligible')]
    eligible_students = eligible['Areas of interest'].value_counts().tolist()
    ineligible = data2[data2['Label'].str.contains('ineligible')]
    ineligible_students = ineligible['Areas of interest'].value_counts().tolist()
    df8 = pd.DataFrame({'Eligible': eligible_students, 'Ineligible': ineligible_students})
    plt.figure(10)
    i = df8.plot(kind='barh', figsize=(15,8),fontsize=10);
    plt.title("Eligibilty for the Applied Technology", fontsize=20)
    i.set_xlabel("Number of Students")
    i.set_ylabel("Areas of interest")
    categories = dataset['Areas of interest'].value_counts().keys().tolist()
    i.set_yticklabels(categories)
    pdf.savefig()  
    plt.close()
    
    # J. Plot the relationship between the year of study, major, and the target variable
    data3 = dataset.filter(['Which-year are you studying in?','Major/Area of Study','Label'])
    plt.figure(11)
    plt.figure(figsize = (10,10))
    j = sns.FacetGrid(data3,hue="Label",size = 10).map(plt.scatter,"Which-year are you studying in?","Major/Area of Study").add_legend();
    plt.xticks(rotation=90)
    plt.title("Relationship between the year of study, major, and the target variable", fontsize=20)
    pdf.savefig()  
    plt.close()
    
