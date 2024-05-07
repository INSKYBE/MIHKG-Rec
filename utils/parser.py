import argparse


def parse_args_MIHKG():
    parser = argparse.ArgumentParser(description="MKG")
    # ===== log ===== #
    parser.add_argument('--desc', type=str, default="", help='EXP description')
    parser.add_argument('--log', action='store_true', default=True, help='log in file or not')
    parser.add_argument('--log_fn', type=str, default=None, help='log file_name')
    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="last-fm",
                        help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument('--model', default="MIHKG", help='use MAE or not')
    # ===== Model Switch ===== #
    parser.add_argument('--mae', action='store_true', default=False, help='use MAE or not')
    parser.add_argument('--ab', type=str, default=None, help='ablation study')
    # ===== Model HPs ===== #
    parser.add_argument('--mae_coef', type=float, default=0.1, help='coefficient for MAE loss')
    parser.add_argument('--mae_msize', type=int, default=256, help='mask size for MAE')
    parser.add_argument('--cl_coef', type=float, default=0.01, help='coefficient for CL loss')
    parser.add_argument('--cl_tau', type=float, default=1.0, help='temperature for CL')
    parser.add_argument('--cl_drop_ratio', type=float, default=0.5, help='drop ratio for CL')

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=int, default=1, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=int, default=1, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=int, default=1, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    # ===== save model ===== #
    parser.add_argument("--save", action='store_true', default=True, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="weights/", help="output directory for model")

    return parser.parse_args()
