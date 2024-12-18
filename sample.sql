-- Insert example API keys

INSERT INTO api_key (id, secret, type, expires, comment) VALUES
('c135f185-786b-4228-8908-88ff14317923', 'df3e6b0bb66ceaadca4f84cbc371fd66e04d20fe51fc414da8d1b84d31d178de', 'LLM', datetime(), 'token1'),
('2c2404d5-7fe1-41e1-8d92-44ed117e2005', 'd8cc7aed3851ac3338fcc15df3b6807b89125837f77a75b9ecb13ed2afe3b49f', 'LLM', NULL, 'token2'),
('28cd867d-5bf8-492a-afd6-5b4c040d53cc', '5d6b091416885eaa91283321b69dc526fc42c97783e4cdfdff7a945e3be1f9ef', 'LLM', '2124-01-01', 'token3');

