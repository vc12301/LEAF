# LEAF
Learning to Extrapolate and Adjust for Forecasting

To reproduce the results, run
```sh
bash run.sh
```

名词：
1. tasks：每个task包含一个train dataset和test dataset，通过rolling的方式下一个task的train dataset为上个task的test dataset。
2. online_step: 根据task训练然后预测然后得到反馈的过程。

我们的onlineEnv环境包含onlineEnv = [(d_0, d_1), (d_1, d_2), (d_2, d_3), ...], 其中d_0为warm up dataset。


我们的code分成几个部分：
1. Baselines文件: 包含baseline算法的code，
    - 所有Latent开头的都是给MetaTSNet用的算法(e.g., LatentMLP, LatentLSTM)其中包含几个部分:
        - g网络: feature extractor, 输入一个X输出一个z。
        - f网络: 我们需要用meta learning生成该参数网络参数（一般为linear 层)。
        - latent：一个低纬度的vector，通过该latent生成f网络参数。
        - param_generator网络：decoder，将latent输入并decode成f网络参数。

    流程为latent -> param_generator -> f的参数, 然后通过g网络抽出的特征z，输出得到预测结果。

    - 所有Naive结尾的都是baseline模型（e.g., MLPNaive, LSTMNaive）其中包含：
        - encoder：对应g网络
        - fine_tune_layer: 对应f网络

        fine_tune模式的话会fine tune fine_tune_layer

2. Online_env文件: 包含环境交互内容。
    - OnlineEnv：整个在线学习环境（一般不需要动)，
    - OnlineModel: 每个online_step的交互（e.g., 每个task如何更新模型？）：
        - OnlineEnvModel （给Naive算法使用）：一个wrapper用于表示每个online step的naive训练流程，该class描述的是最基础的训练流程，给定X_train, y_train我们如何minimize一个loss。
        - OnlineMeta2Model (给MetaTS2Net设计)：一个wrapper用于表示每个online step的MetaTS2Net训练流程，warm_up结束之后在train上adapt，然后在y上通过update_meta_model函数更新meta参数。
    
    - Train_cls: 一整个online过程的调度(e.g., 每个task怎么分train test，meta_train, meta_test怎么分)：
        - OnlineEnvTrain：定义了不同模式(fine_tune, naive, meta_finetune)的交互方式，为每个OnlineModel(OnlineEnvModel, OnlineMeta2Model)提供task以及记录预测值和准确率。
    
3. meta2net文件：包含了我们的MetaTS2Net的方法。
    - target_model：某个Latent模型(e.g., LatentMLP, LatentLSTM)。
    - extrapolation_network: 预测得到下个dataset的latent。
    - meta_loss：一个unsupervised_loss将输入X进行一步更新。
    - forward_latent: 通过extrapolation_network输入latent_queue预测得到下个阶段的latent，然后将该latent置入target_model.latent。
    - sample_adapt: 将X结合latent通过meta_loss更新至X`。
    - adapt：得到当前train_dataset的latent并放入latent_queue用于forward_latent。


整体流程 (MetaTS2Net)：
1. OnlineEnvTrain调用某个mode （e.g., fine_tune, naive, meta_finetune）, 然后和onlineEnv交互得到当前online_step的task, 将该task中的train_dataset输入OnlineMeta2Model.
2. OnlineMeta2Model通过train_dataset进行adapt得到当前train_dataset的latent并放入forward_latent。
3. OnlineEnvTrain调用OnlineMeta2Model的预测得到test_dataset的预测值，计算score并record。
4. OnlineEnvTrain调用OnlineMeta2Model的update_meta_model函数，输入test_dataset从而更新meta_model参数。


Q1: 如果我需要新增一个算法，但是每个online_step流程和OnlineEnvModel, OnlineMeta2Model不一样怎么办?
A: 按照OnlineEnvModel, OnlineMeta2Model的格式重新写一个，然后在配置中的env_model参数设为你的Model，必要的话需要新增一个OnlineEnvTrain中的mode。

Q2: 如果我需要新增一个baseline算法怎么办？
A： 按照Latent和Naive算法的结构重新写一个算法。

Q3：如果我想修改meta model更新过程怎么办？
A： 修改OnlineMeta2Model.update_meta_model

Q4：如果我想修改warm_up或者adapt过程怎么办？
A： 修改OnlineMeta2Model.fit

Q5: 如何配置参数？
A: 每个dataset和算法的配置都在main.py中fetch_train_cls中有定义，如需要新增或者修改需要修改其中对应的数值。

