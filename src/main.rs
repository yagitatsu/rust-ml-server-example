use axum::routing::{get, post};
use axum::{extract, response, Json, Router};
use gbdt::config::Config;
use gbdt::decision_tree::{Data, DataVec, PredVec};
use gbdt::gradient_boost::GBDT;
use gbdt::input::{load, InputFormat};
use once_cell::sync::OnceCell;
use serde::Deserialize;
use serde_json::{json, Value};
use std::net::SocketAddr;

static MODEL: OnceCell<GBDT> = OnceCell::new();

#[tokio::main]
async fn main() {
    let model = GBDT::load_model("gbdt.model").expect("failed to load the model");
    MODEL.set(model);

    // build our application with a single route
    let app = Router::new()
        .route("/", post(root))
        .route("/workflow/train", get(training));

    // run it with hyper on localhost:3000
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

#[derive(Deserialize)]
struct Payload {
    year: f32,
    month: f32,
    season: f32,
}

// which calls one of these handlers
async fn root(extract::Json(payload): extract::Json<Payload>) -> Json<Value> {
    if payload.year == 0.0f32 && payload.month == 0.0f32 && payload.season == 0.0f32 {
        return response::Json(json!({ "error": "payload is empty" }));
    }
    let test_data = Data::new_test_data(vec![payload.year, payload.month, payload.season], None);

    let model = MODEL.get().expect("model is not initialized");

    let predicted: PredVec = model.predict(&vec![test_data]);

    return Json(json!({ "data": predicted }));
}

async fn training() {
    let mut cfg = Config::new();
    cfg.set_feature_size(3);
    cfg.set_max_depth(3);

    let input_file = "./input.csv";
    let mut fmt = InputFormat::csv_format();
    fmt.header = true;
    fmt.set_feature_size(3);
    fmt.set_label_index(3);
    let mut input_data: DataVec = load(input_file, fmt).expect("failed to load input data");

    // train and save model
    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&mut input_data);
    gbdt.save_model("gbdt.model")
        .expect("failed to save the model");

    println!("training done");
}
