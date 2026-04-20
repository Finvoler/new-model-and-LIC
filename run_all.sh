# 涓夋ā鍨嬮『搴忚缁冭剼鏈?(Fourier -> Gaussian -> LIC)
# 鍏辩敤鍙傛暟: dim=64, layers=3, lr=0.001, decay=1e-4, batch=4096, epochs=100

$commonArgs = @(
    "--data_path", "./data/taobao_subset_large",
    "--min_interactions", "15",
    "--latent_dim_rec", "64",
    "--lightGCN_n_layers", "3",
    "--epochs", "100",
    "--bpr_batch_size", "4096",
    "--test_u_batch_size", "512",
    "--topks", "[10,20]",
    "--test_interval", "5",
    "--lr", "0.001",
    "--decay", "1e-4",
    "--device", "cuda"
)

Write-Host "=" * 60
Write-Host "[1/3] Fourier 妯″瀷璁粌"
Write-Host "=" * 60
python main.py @commonArgs `
    --temporal_model fourier `
    --n_clusters 4 --fourier_k 3 --tau 0.5 `
    --entropy_weight 0.01 --fusion_mode add `
    2>&1 | Tee-Object -FilePath "log_fourier_large.txt"

Write-Host ""
Write-Host "=" * 60
Write-Host "[2/3] Gaussian Interest Clock 妯″瀷璁粌"
Write-Host "=" * 60
python main.py @commonArgs `
    --temporal_model gaussian `
    --clock_emb_dim 64 `
    --clock_gaussian_mu 0.0 --clock_gaussian_sigma 1.0 `
    --time_diff_alpha 8.0 `
    2>&1 | Tee-Object -FilePath "log_gaussian_large.txt"

Write-Host ""
Write-Host "=" * 60
Write-Host "[3/3] LIC (Long-term Interest Clock) 妯″瀷璁粌"
Write-Host "=" * 60
python main.py @commonArgs `
    --temporal_model lic `
    --lic_top_k 100 --lic_n_heads 4 --lic_d 32 --lic_alpha 1.0 `
    2>&1 | Tee-Object -FilePath "log_lic_large.txt"

Write-Host ""
Write-Host "=" * 60
Write-Host "鎵€鏈夋ā鍨嬭缁冨畬鎴愶紒鏃ュ織鏂囦欢:"
Write-Host "  Fourier:  log_fourier_large.txt"
Write-Host "  Gaussian: log_gaussian_large.txt"
Write-Host "  LIC:      log_lic_large.txt"
Write-Host "=" * 60
