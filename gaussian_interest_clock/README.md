# Gaussian Interest Clock on LightGCN (Taobao Adaptation)

## 鐩爣
鍦ㄤ繚鎸?LightGCN 楠ㄥ共鍜岃緭鍏ユ帴鍙ｄ笉鍙樼殑鍓嶆彁涓嬶紝澶嶇幇 Interest Clock 鐨勬牳蹇冩満鍒讹細
1) 鍏堟瀯寤虹敤鎴锋寜灏忔椂鐨勪釜鎬у寲鍏磋叮鏃堕挓锛?
2) 鍐嶅 24 灏忔椂鐗瑰緛鍋氶珮鏂钩婊戣仛鍚堬紱
3) 鐢ㄦ椂閽熷悜閲忎笌鐩爣鐗╁搧绫诲埆鍚戦噺浜ゅ弶鎵撳垎銆?

## 娣樺疂鐗瑰緛瀵归綈
- 璁烘枃涓殑 genre/mood/lang 鏇挎崲涓哄崟涓€ Category銆?
- 涓?Category 缁存姢鐙珛 embedding銆?

## 绂荤嚎棰勫鐞嗭紙涓嶅湪 forward 缁熻锛?
- 璁粌闃舵鏍锋湰鎸?24 灏忔椂妗剁粺璁＄敤鎴风偣鍑荤被鍒娆°€?
- 姣忎釜鐢ㄦ埛姣忓皬鏃堕€?Top-3 Category銆?
- 棰勫瓨寮犻噺锛歶ser_hour_top3_categories锛屽舰鐘?[num_users, 24, 3]銆?

## 鍦ㄧ嚎鑱氬悎涓庢墦鍒?
- 鏍规嵁璇锋眰鏃堕棿 thetas 璁＄畻鍒?24 灏忔椂妗剁殑寰幆璺濈銆?
- 閫氳繃楂樻柉鍑芥暟鐢熸垚 24 缁存潈閲嶅苟褰掍竴鍖栥€?
- 鍙栫敤鎴峰湪 24 灏忔椂鐨?Top-3 绫诲埆 embedding锛屾瘡灏忔椂鍋?mean-pooling銆?
- 鐢ㄩ珮鏂潈閲嶅姞鏉冩眰鍜屽緱鍒?v_clock銆?
- 鐩爣鐗╁搧鎸?item->category 鍙?v_target_category銆?
- 鏈€缁堝垎鏁帮細
  score = dot(user_lightgcn, item_lightgcn) + alpha * dot(v_clock, v_target_category)

## 杩愯鏂瑰紡
- 鍌呴噷鍙剁増鏈細
  python main.py --temporal_model fourier
- Interest Clock 楂樻柉鐗堟湰锛?
  python main.py --temporal_model gaussian

## 鍏抽敭鍙傛暟
- --clock_emb_dim: Category embedding 缁村害
- --clock_gaussian_mu: 楂樻柉骞虫粦鍧囧€硷紙璁烘枃缁忛獙鍊?0锛?
- --clock_gaussian_sigma: 楂樻柉骞虫粦鏂瑰樊灏哄害锛堣鏂囩粡楠屽€?1锛?
- --time_diff_alpha: 鏃堕挓浜ゅ弶椤规潈閲?alpha
