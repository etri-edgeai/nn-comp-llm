import argparse

def parse_args(run_type='terminal'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--gpuid", default='0', type=str,
                        help="id(s) for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--student_model", default=None, type=str, # required=True,
                        help="The pretrained model dir.")
    parser.add_argument("--teacher_model", default=None, type=str, # required=False,
                        help="The pretrained model dir.")
    parser.add_argument("--task_name", default=None, type=str, # required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=2e-05, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument('--eval_step', type=int, default=int(1e6))
    parser.add_argument('--mode', type=str, default="FT") 
    parser.add_argument('--scratch', action="store_true",
                        help="Whether to start training from scratch")
    parser.add_argument('--pkd_initialize', action="store_true",
                        help="Whether to use initialization method in PKD-BERT")
    parser.add_argument('--tinybert_aug', action="store_true",
                        help="Whether to use augmentation method in TinyBERT")
    parser.add_argument('--use_mixcons', action="store_true",
                        help="Whether to use consistency mixup")

    # For HintonKD or PKD
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--temperature", default=2.0, type=float, 
                        help="Distillation temperature. Only for distillation.")
    parser.add_argument("--beta", default=10.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")

    # For TinyBERT
    parser.add_argument("--hidden_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--attention_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--embedding_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--logit_distill", action="store_true", 
                        help="Distillation objective for CLS. Only for distillation.")
    parser.add_argument("--distill_object", type=str, default="",
                        help="Distillation objective for ILD. Only for distillation.")
    parser.add_argument("--hidden_l2l", action="store_true",
                        help="Whether to use Layer-to-Layer distillation with hidden representation")

    # For MiniLM
    parser.add_argument("--attention_l2l", action="store_true",
                        help="Whether to use Layer-to-Layer distillation with self-attention")
    parser.add_argument("--attention_l2i", action="store_true")
    parser.add_argument("--l2l", action="store_true",
                        help="Whether to use Layer-to-Layer distillation with relational knowledge")

    # For BERT-EMD
    parser.add_argument("--use_att", action="store_true",
                        help="Whether to use attention-based distillation")
    parser.add_argument("--use_rep", action="store_true",
                        help="Whether to use hidden-based distillation")
    parser.add_argument("--update_weight", action="store_true",
                        help="Whether to use weight update with cost attention")
    parser.add_argument("--embedding_emd", action="store_true",
                        help="Whether to use emd to embedding transfer")
    parser.add_argument("--seperate", action="store_true",
                        help="Whether to use averaged weights for attention and hidden")
    parser.add_argument("--T_emd", default=1.0, type=float,
                        help="temperature for knowledge distillation")

    # For Ours
    parser.add_argument("--refine_model", default=None, type=str,
                        help="self-teacher model for refinement")
    parser.add_argument("--decay", type=float, default=0.999,
                        help="decay for exponential moving average")
    parser.add_argument("--use_ema", action="store_true",
                        help="Whether to use exponential moving average for updating refine model")
    parser.add_argument("--use_mixup", action="store_true",
                        help="Whether to use mixup augmentation")
    parser.add_argument("--use_ic_att", action="store_true",
                        help="Whether to use inter consistency")
    parser.add_argument("--use_ic_rep", action="store_true",
                        help="Whether to use inter consistency")
    parser.add_argument("--w_ic", type=float, default=0.1,
                        help="Coefficient for inter consistency")

    # Few-samples
    parser.add_argument("--downsample", action="store_true",
                        help="Whether to use downsampling for generating few-sample dataset")

    # Label-Distribution-Aware Setup
    parser.add_argument("--mislabel", action="store_true",
                        help="Whether to inject label noise")
    parser.add_argument("--mislabel_ratio", type=float, default=None)
    parser.add_argument("--imbalance_ratio", type=float, default=1.0)

    parser.add_argument("--joint_train", action="store_true",
                        help="Whether to use joint train with ILD and KD")
    parser.add_argument("--w_kd", type=float, default=0.0,
                        help="coefficient for kd weights in joint training")

    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_final_model", action="store_true")
    
    if run_type=='terminal':
        args = parser.parse_args()
    elif run_type=='jupyter':
        args = parser.parse_args(args=[])
    else:
        raise NotImplementedError
        
    return args