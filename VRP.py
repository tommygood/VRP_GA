import sys
import math
import random
import numpy as np
from numpy import sin, cos, arccos, pi


class GA:
    def __init__(self, total_p, city, dis_list, size, gen_rate, mut_rate,car,car_pivot):
        # 目前採用方法分別為: 公開賽法 / 雙點交配 / 交換突變
        self.total_p = total_p # 總共人口數
        self.city = city # 城市數量
        self.dis_list = dis_list # 每個點之間的距離
        self.size = size # 族群大小
        self.gen_rate = gen_rate # 交配率
        self.mut_rate = mut_rate # 變異率
        self.car = car # 車子的數量      
        self.car_pivot = car_pivot # 每一台車平均下要跑幾公里的標準          
        self.population = self.init_population() # 產生最初族群
        self.answer = [[0]*(self.city+self.car-2),sys.maxsize] # 存放最佳解
        
    def init_population(self): # 初始化族群
        population = []
        for index in range(self.size):
            chromosome = list(range(2,self.city+1))
            random.shuffle(chromosome) # 將列表內元素隨機重新排列 (為 random 模塊提供的函數之一)
            chromosome = self.init_cut(chromosome) # 加入每段路線的切點, 為一條基因
            # 一個個體內容依序為: 在族群內序號，基因，適應度(分數)，是否當過父母
            population.append([index,chromosome,self.Fitness_distance(self.cut_road_list(chromosome)),False])
        return population
    
    def init_cut(self,gene): # 按照車輛數將路徑切成 n 段
        # random_num: 隨機 0-1 數字，代表該車要跑全部路線的權重
        # cut_list: 每台車要跑的城市數量
        # last: 最後一段路的長度 (減 1 是因為第一個城市 (起點) 不用被排)
        random_num,cut_list,last = [],[0 for i in range(self.car)],self.city-1
        
        """
        # 原本寫法會有車子是負數個城市, 所以前面車子會跑超過城市的數量 的問題
        cut_list = []
        for _ in range(self.car):
            random_num.append(random.random()) # 每段隨機長度
        
        # 前 n-1 段
        for i in range(self.car):
            # 依照每台車隨機比例, 放要跑的城市數量
            cut_list.append(round(self.city*random_num[i]/sum(random_num))) 
            last -= cut_list[i]
        cut_list.append(last) # 最後一段
        #print(cut_list)
        #if sum(cut_list) <= self.city-2 :
            #print(sum(cut_list))
        """
        for i in range(last) :
            random_car = random.randint(0, len(cut_list)-1)
            cut_list[random_car] += 1
        #print(cut_list, sum(cut_list))          

        cut_point = self.city # 切斷點為加入比城市代號大的數字表示
        for i in range(len(cut_list)):
            index = 0 # 在第 index 個插入切斷點
            for a in range(i):
                index += cut_list[a] # 用每台車要負責幾個城市算 index
            index += i
            gene.insert(index,cut_point) # 在第 index 個插入切斷點
            cut_point += 1 # 切斷點代表數字增加
        del gene[0]
        return gene
 
    def cut_road_list(self,gene): # 將每段路線切成一個陣列
        result,temp = [],[] 
        for x in gene:
            if x > self.city: #　如果當前數字為切斷點則將目前陣列加入 result 中，創建一新證列記錄下段陣列
                result.append(temp)
                temp = []
            else:
                temp.append(x)
        if temp: # 當 temp 不為空值時才加入
            result.append(temp)
        return result

    def Fitness_distance(self,gene_list): # 計算分數 (總距離)
        score = 0
        over_limit_car = 0
        each_route_score = []
        for gene in gene_list: # 同一個染色體的基因, 由多條路線(一台車負責一條路線)的多個 list 組成
            road_score = 0 # 該路線總長
            if (gene != []): # 當該路線需要車子出發
                for i in range(len(gene)-1):
                    # 抓取 gene[i] 和 gene[i+1] 之間的歐幾里德距離，並將其加到 road_score 中
                    a,b = sorted([gene[i],gene[i+1]])
                    road_score += self.dis_list[a-1][b-a-1]
                # 抓取起點到第一個城市，和最後一個城市會到起點的距離，並將其加到 road_score 中
                a,b,c = 1,gene[0],gene[-1]
                road_score += self.dis_list[a-1][b-a-1]
                road_score += self.dis_list[a-1][c-a-1]
                each_route_score.append(road_score)
                
            # limit: 限制條件(每台車不能跑超過多少公里) / weight: 權重，指的是超過懲罰應占比多少
            limit = self.car_pivot # 作為其他參數的指標
            weight, min_limit, weight_over_limit_car = limit*80, limit/1.75, limit/6
            if road_score > limit: # 單一路線跑太多公里要懲罰
                over_limit_car += weight_over_limit_car # 但是越多台車跑超過, 懲罰的越少(因可能是 limit 設的不好讓大家都超過)
                score += (road_score + ((road_score-limit)*weight / over_limit_car))
            elif road_score < min_limit : # 單一路線跑太少公里也要懲罰
                score += (road_score + ((limit-road_score)*weight))
            else : # 沒有跑太少也沒有跑太多公里
                score += road_score
            # 懲罰單一路線跑太多或太少城市
            city_limit = round((self.city - 1) / self.car) # 平均下, 每個 car 要跑多少 city
            w_city = limit * 10
            if len(gene) > city_limit * 1.25 : # 跑太多城市(超過平均)
                score += (len(gene) - city_limit) * w_city
            elif len(gene) < city_limit/2 : # 跑太少城市(低於平均的一半)
                score += (len(gene) - city_limit) * w_city * 2 # 跑太少的懲罰要比跑太多高
        
        # 懲罰標準差太大的公里數
        each_route_score = np.array(each_route_score) 
        std = np.std(each_route_score, ddof=1) # 算標準差
        std_limit = 3 # 標準差限制, 越高, 車子公里數差距會越大
        if std > std_limit :
            score += (std-std_limit) * weight * 500 # 超過標準差就懲罰爆, 因為讓公里數平均很重要
        return score
    
    def Selection_Tournament(self): # 使用公開賽法 但這樣會選到重覆的染色體
        new_population = []
        for index in range(self.size):
            player = random.choices(self.population, k=8) # 從族群中依據先前製作之機率列表隨機挑選數個, 這邊 k 應要改成基於 city
            fit_list = [gene[2] for gene in player] # 被挑選的染色體的 fitness
            winner = player[fit_list.index(min(fit_list))] # 選 fitness 最小的
            new_population.append([index,winner[1],winner[2],False])    
        return new_population
    
    def Choose_parent(self): # 選擇兩條染色體的基因當作父母
        remaining_population = [gene for gene in self.population if gene[3] == False] # 挑初沒被選過當父母的
        choose = random.choices(remaining_population, k=2) # 從族群中依據先前製作之機率列表隨機挑選兩個
        self.population[choose[0][0]][3],self.population[choose[1][0]][3] = True,True  # 標記已有選過當作父母
        return choose
    
    def Crossover(self, parent1, parent2): # 交配法，使用雙點交配法 (PMX)
        gene_len = self.city+self.car-2
        child = [0]*gene_len
        start, end = sorted(random.sample(range(gene_len), 2)) # 隨機挑選切兩刀的位置
        for i in range(start, end+1): # 先讓前段等於父母一號
            child[i] = parent1[1][i]
        j = end + 1
        for i in range(gene_len):
            if parent2[1][i] not in child: # 如果這個數字還沒在child裡面
                if j == gene_len: # 如果已經放到尾則從頭繼續檢查
                    j = 0
                child[j] = parent2[1][i] # 放入child
                j += 1 # child 要放的位置輪到下一個
        return child

    def Mutation(self, chromosome): #　變異法，選擇交換突變
        i, j = random.sample(range(self.city), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome
        
    def Evolve(self): # 演化過程
        for _ in range(math.ceil(self.total_p/self.size)): # 總共有幾代人口      
            new_population = []
            self.population = self.Selection_Tournament() # 天擇
            for i in range(0,self.size,2): # 交配
                parent1, parent2 = self.Choose_parent() # 先隨機從族群中選兩條沒被選過當父母的染色體的基因作為父母
                if random.random() < self.gen_rate: # 如果有交配
                    child1,child2 = self.Crossover(parent1, parent2),self.Crossover(parent2, parent1) # 產生新子代染色體的基因
                    # 新的 population 加入新子代染色體
                    new_population.append([i,child1,self.Fitness_distance(self.cut_road_list(child1)),False])
                    new_population.append([i+1,child2,self.Fitness_distance(self.cut_road_list(child2)),False])
                else: # 若無交配，則填入原本基因
                    new_population.append([i,parent1[1],parent1[2],False])
                    new_population.append([i+1,parent2[1],parent2[2],False])
            
            for i in range(self.size): # 突變
                if random.random() < self.mut_rate:
                    gene = self.Mutation(new_population[i][1])
                    new_population[i] = [i,gene,self.Fitness_distance(self.cut_road_list(gene)),False]
                   
            self.population = new_population # 更新人口
            f_list = [gene[2] for gene in self.population] # 找距離最小的最佳解做比對
            best_fit = min(f_list)
            if ( best_fit < self.answer[1]): # 如果更小則替換
                self.answer = [self.population[f_list.index(best_fit)][1],best_fit]
        return self.answer
      
def main():
    # time: 重複執行GA的次數 / total_p: 總共人口數 / size: 族群大小
    # gen_rate: 交配率      / mut_rate: 變異率    / file_name: 要使用座標之檔案名稱 / car: 最多可派出的車子數
    #time,total_p,size,gen_rate,mut_rate,file_name,car = 10,10000,100,0.92,0.12,'Berlin52.txt',5  
    time,total_p,size,gen_rate,mut_rate,file_name,car = 30,10000,100,0.92,0.12,'delivery.txt',15
    with open(file_name, 'r') as f: 
        city = int(f.readline().strip()) # 從檔案第一行抓取城市數量
        # 使用列表解析式將每行的坐標轉換為元組，並將所有元組存儲到列表 city_coordinate 中
        city_coordinate = [tuple(map(float, line.strip().split()[1:])) for line in f.readlines()] 

    print("\n======= 目前測試參數數據 =======")
    print("file_name : ",file_name)
    print("Time     =",time,"   city     =",city,
          "\ntotal_p  =",total_p,"gen_rate =",gen_rate,
          "\nsize     =",size,"  mut_rate =",mut_rate)
    print("================================")

    ans_list,fit_list,dis_list = [],[],init_dis_list(city,city_coordinate) # 儲存結果之陣列
    car_pivot = each_car_pivot(dis_list[0], car) # 每一台車平均下要跑幾公里的標準
    for i in range(time):
        print("Exam :",i+1,"time")
        test = GA(total_p,city,dis_list,size,gen_rate,mut_rate,car,car_pivot) # 人口數, 城市數量, 城市之間距離, 族群大小, 交配率, 變異率, 車子數
        ans = test.Evolve() # 回傳這一代的最好染色體的基因
        ans_list.append(test.cut_road_list(ans[0])) # 將每段路線切成一個陣列
        fit_list.append(ans[1]) # 這個染色體的 fitness
        print("GBEST FIT =",ans[1])
        print("================================")
   
    mean_value = sum(fit_list) / len(fit_list)
    std_value = math.sqrt(sum((x - mean_value) ** 2 for x in fit_list) / (len(fit_list) - 1))
    print("GBEST Fitness      = ", round(min(fit_list),4))
    print("Average            = ", round(mean_value,4))
    print("Standart Deviation = ", round(std_value,6))
    
    best_answer,i,total_dis = ans_list[fit_list.index(min(fit_list))],1,0
    print("\n================== GBEST tour detailed ==================")
    for road_list in best_answer:
        if road_list != []:
            dis = count_distance(dis_list,road_list)
            print("Car "+str(i)+" Route    :",*road_list)
            print("      Distance :",dis)
            print("      Route counts :", len(road_list))
            print("==========================================================")
            i += 1
            total_dis += dis
    print("Total Distance =",total_dis) 

def each_car_pivot(city_dis, car) :
    avg_each_city_dis = sum(city_dis) / len(city_dis)
    avg_car_city = len(city_dis) / car
    return avg_each_city_dis * avg_car_city

def init_dis_list(city,city_coordinate): # 計算每個點之間的距離 (算分數時即可直接呼叫兩點間距離而省去重複計算的時間)
    list = []
    for i in range(city-1): # 目前城市
        dis = []
        for j in range(i+1,city): # 下一個城市 
            dis.append(euclidean_distance(city_coordinate[i], city_coordinate[j]))
        list.append(dis)
    return list
    
def euclidean_distance(a, b): # 計算 a 和 b 之間的歐幾里德距離 (兩點直線距離)
    #return math.sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))])) # 算兩點直線距離
    return getDistanceBetweenPointsNew(a[0],a[1],b[0],b[1]) # 算經緯度直線距離

def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees

def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians

def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2):
    theta = longitude1 - longitude2
    distance = 60 * 1.1515 * rad2deg(
        arccos(
            (sin(deg2rad(latitude1)) * sin(deg2rad(latitude2))) + 
            (cos(deg2rad(latitude1)) * cos(deg2rad(latitude2)) * cos(deg2rad(theta)))
        )
    )
    return round(distance * 1.609344, 2)

def count_distance(dis_list,gene): # 計算距離
        score = 0
        for i in range(len(gene)-1):
            # 抓取 gene[i] 和 gene[i+1] 之間的歐幾里德距離，並將其加到 score 中
            a,b = sorted([gene[i],gene[i+1]])
            score += dis_list[a-1][b-a-1]
        # 抓取首尾城市之間的歐幾里德距離，並將其加到 score 中
        a,b,c = 1,gene[0],gene[-1]
        score += dis_list[a-1][b-a-1]
        score += dis_list[a-1][c-a-1]
        return score

if __name__ == '__main__':
    main()