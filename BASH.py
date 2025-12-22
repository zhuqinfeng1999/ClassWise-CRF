import os
import time
import subprocess
import logging

# 配置日志
log_file_path = r"/ZQFSSD/crf/script_execution_uavid.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path)]
)

# 脚本列表
scripts = [
    #r"/ZQFSSD/crf/NPY_NOFUSE/crf_convnext_bayesian_vaihingen.py",
    r"/ZQFSSD/crf/NPY_NOFUSE/crf_swin_bayesian_vaihingen.py",
    r"/ZQFSSD/crf/NPY_NOFUSE/crf_vmamba_bayesian_vaihingen.py",

    r"/ZQFSSD/crf/NPY_SIMPLE_FUSE/convnext_swin_bayesian_vaihingen.py",
    r"/ZQFSSD/crf/NPY_SIMPLE_FUSE/convnext_vmamba_bayesian_vaihingen.py",
    r"/ZQFSSD/crf/NPY_SIMPLE_FUSE/swin_vmamba_bayesian_vaihingen.py",

    r"/ZQFSSD/crf/NPY_WEIGHT_FUSE/convnext_swin_bayesian_vaihingen.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_FUSE/convnext_vmamba_bayesian_vaihingen.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_FUSE/swin_vmamba_bayesian_vaihingen.py",

    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/convnext_swin_bayesian_vaihingen_e1.5.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/convnext_swin_bayesian_vaihingen_e2.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/convnext_swin_bayesian_vaihingen_e2.5.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/convnext_swin_bayesian_vaihingen_e3.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/convnext_vmamba_bayesian_vaihingen_e1.5.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/convnext_vmamba_bayesian_vaihingen_e2.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/convnext_vmamba_bayesian_vaihingen_e2.5.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/convnext_vmamba_bayesian_vaihingen_e3.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/swin_vmamba_bayesian_vaihingen_e1.5.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/swin_vmamba_bayesian_vaihingen_e2.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/swin_vmamba_bayesian_vaihingen_e2.5.py",
    r"/ZQFSSD/crf/NPY_WEIGHT_E_FUSE/swin_vmamba_bayesian_vaihingen_e3.py",
]

# 执行脚本
for script in scripts:
    logging.info(f"开始执行脚本: {script}")
    start_time = time.time()
    try:
        subprocess.run(["python", script], check=True)
        logging.info(f"脚本 {script} 执行成功")
    except subprocess.CalledProcessError as e:
        logging.error(f"脚本 {script} 执行失败: {e}")
    end_time = time.time()
    logging.info(f"脚本 {script} 执行耗时: {end_time - start_time:.2f} 秒")

logging.info("所有脚本执行完毕")

