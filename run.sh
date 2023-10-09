# # Meta LSTM
python main.py --datasets bus_1 --algorithms meta_lstm  --mode meta_finetune  --save_dir_name "meta_lstm_bus_1" --rep 3
python main.py --datasets bus_2 --algorithms meta_lstm  --mode meta_finetune  --save_dir_name "meta_lstm_bus_2" --rep 3
python main.py --datasets bus_3 --algorithms meta_lstm  --mode meta_finetune  --save_dir_name "meta_lstm_bus_3" --rep 3
python main.py --datasets etth2 --algorithms meta_lstm --mode meta_finetune  --save_dir_name "meta_lstm_etth2" --rep 3
python main.py --datasets ettm1 --algorithms meta_lstm --mode meta_finetune  --save_dir_name "meta_lstm_ettm1" --rep 3
python main.py --datasets ECL --algorithms meta_lstm  --mode meta_finetune  --save_dir_name "meta_lstm_ecl" --rep 3

# # Meta Dlinear
python main.py --datasets bus_1 --algorithms meta_dlinear  --mode meta_finetune  --save_dir_name "meta_dlinear_bus_1" --rep 3
python main.py --datasets bus_2 --algorithms meta_dlinear  --mode meta_finetune  --save_dir_name "meta_dlinear_bus_2" --rep 3
python main.py --datasets bus_3 --algorithms meta_dlinear  --mode meta_finetune  --save_dir_name "meta_dlinear_bus_3" --rep 3
python main.py --datasets etth2 --algorithms meta_dlinear --mode meta_finetune  --save_dir_name "meta_dlinear_etth2" --rep 3
python main.py --datasets ettm1 --algorithms meta_dlinear --mode meta_finetune  --save_dir_name "meta_dlinear_ettm1" --rep 3
python main.py --datasets ECL --algorithms meta_dlinear --mode meta_finetune  --save_dir_name "meta_dlinear_ecl" --rep 3



# # Meta Patch
python main.py --datasets bus_1 --algorithms meta_patch  --mode meta_finetune  --save_dir_name "meta_patch_bus_1" --rep 3
python main.py --datasets bus_2 --algorithms meta_patch  --mode meta_finetune  --save_dir_name "meta_patch_bus_2" --rep 3
python main.py --datasets bus_3 --algorithms meta_patch  --mode meta_finetune  --save_dir_name "meta_patch_bus_3" --rep 3
python main.py --datasets etth2 --algorithms meta_patch --mode meta_finetune  --save_dir_name "meta_patch_etth2" --rep 3
python main.py --datasets ettm1 --algorithms meta_patch --mode meta_finetune  --save_dir_name "meta_patch_ettm1" --rep 3
python main.py --datasets ECL --algorithms meta_patch --mode meta_finetune  --save_dir_name "meta_patch_ecl" --rep 3


# # Meta TCN
python main.py --datasets bus_1 --algorithms meta_tcn  --mode meta_finetune  --save_dir_name "meta_tcn_bus_1" --rep 3
python main.py --datasets bus_2 --algorithms meta_tcn  --mode meta_finetune  --save_dir_name "meta_tcn_bus_2" --rep 3
python main.py --datasets bus_3 --algorithms meta_tcn  --mode meta_finetune  --save_dir_name "meta_tcn_bus_3" --rep 3
python main.py --datasets etth2 --algorithms meta_tcn --mode meta_finetune  --save_dir_name "meta_tcn_etth2" --rep 3
python main.py --datasets ettm1 --algorithms meta_tcn --mode meta_finetune  --save_dir_name "meta_tcn_ettm1" --rep 3
python main.py --datasets ECL --algorithms meta_tcn --mode meta_finetune  --save_dir_name "meta_tcn_ecl" --rep 3