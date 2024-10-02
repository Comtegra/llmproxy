// Insert example API keys

cgc = db.getSiblingDB("cgc");

cgc.api_keys.insertOne({access_level: "COMPLETION", user_id: "user1", date_expiry: new Date(),
    secret: "df3e6b0bb66ceaadca4f84cbc371fd66e04d20fe51fc414da8d1b84d31d178de"}); // "token1"

cgc.api_keys.insertOne({access_level: "COMPLETION", user_id: "user2",date_expiry: null,
    secret: "d8cc7aed3851ac3338fcc15df3b6807b89125837f77a75b9ecb13ed2afe3b49f"}); // "token2"

cgc.api_keys.insertOne({access_level: "COMPLETION", user_id: "user3", date_expiry: new Date("2124-01-01"),
    secret: "5d6b091416885eaa91283321b69dc526fc42c97783e4cdfdff7a945e3be1f9ef"}); // "token3"
