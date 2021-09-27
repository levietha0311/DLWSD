# DLWSD - A Proactive Method of the Webshell Detection and Prevention based on Deep Traffic Analysis


Perform hybrid approach that is combination of rule-based IDPS and Deep Neural Network for webshell detection based on deep HTTP traffic analysis.

For the paper "A Proactive Method of the Webshell Detection and Prevention based on Deep Traffic Analysis"!

In order to validate and evaluate the effectiveness of the DLWSD method, we use two sub-datasets

1. The first sub-dataset is built by deploying a 2-component testbed system Webserver-Network and Attack-Network. We use several craws website tools to create the normally HTTP traffic as legal clients.  we also simulate Webshell attacks by using Kali Linux to upload and execute webshell for creating intrusion traffic. Suricata is used as a packet capturing, HTTP filtering tool and saves network traffic into pcap files. 

2. The second sub-dataset is a well-known and reliable CSE-CIC-IDS2018 dataset that is published by Canadian Institute for Cybersecurity.

From these two sub-datasets, we constitute a dataset for testing. The dataset contains two parts: training and testing at the ratio of 7:3.

For full access to all the source code and datasets we use, please download by following the link: 
https://drive.google.com/drive/folders/1gBN-YXZsdfhh0Tp2OHykB3H8NtLXzzZ5?usp=sharing


With the contributions of the authors:

Ha V. Le - levietha@chinhphu.vn - Department of Information Systems, VNU University of Engineering and Technology, Hanoi, Vietnam

Hanh P. Du - Department of Information Systems, VNU University of Engineering and Technology, Hanoi, Vietnam

Cuong N. Nguyen - cuongnn.hvan@gmail.com - Department of Network Security, Ministry of Police, Hanoi, Vietnam 

Long V. Hoang - longhv08@gmail.com - Faculty of Information Technology, University of Technology-Logistics of Public Security, Bac Ninh, Vietnam

Hoa N. Nguyen - Corresponding author - hoa.nguyen@vnu.edu.vn - Department of Information Systems, VNU University of Engineering and Technology, Hanoi, Vietnam
