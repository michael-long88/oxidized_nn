use std::{fs::File, io::BufReader};
use csv::Reader;
use serde::Deserialize;

use oxidized_nn::*;


/// A struct to hold the data from the wheat.csv file.
#[derive(Deserialize, Clone, Debug)]
pub struct Records {
    area: Vec<f64>,
    perimeter: Vec<f64>,
    compactness: Vec<f64>,
    kernel_length: Vec<f64>,
    kernel_width: Vec<f64>,
    asymmetry_coefficient: Vec<f64>,
    kernel_groove_length: Vec<f64>,
    type_of_wheat: Vec<i8>,
}

/// Reads the wheat.csv file and returns a vector of Records.
pub fn read_wheat_csv(filename: &str) -> Records {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut records = Records {
        area: vec![],
        perimeter: vec![],
        compactness: vec![],
        kernel_length: vec![],
        kernel_width: vec![],
        asymmetry_coefficient: vec![],
        kernel_groove_length: vec![],
        type_of_wheat: vec![],
    };
    

    for record in Reader::from_reader(reader).into_records() {
        let record = record.unwrap();

        records.area.push(record[1].parse::<f64>().unwrap());
        records.perimeter.push(record[2].parse::<f64>().unwrap());
        records.compactness.push(record[3].parse::<f64>().unwrap());
        records.kernel_length.push(record[4].parse::<f64>().unwrap());
        records.kernel_width.push(record[5].parse::<f64>().unwrap());
        records.asymmetry_coefficient.push(record[6].parse::<f64>().unwrap());
        records.kernel_groove_length.push(record[7].parse::<f64>().unwrap());
        records.type_of_wheat.push(record[8].parse::<i8>().unwrap());
    }

    records
}

/// Calculates the min and max values for each column in the dataset.
pub fn get_min_max(records: &Records) -> Vec<(f64, f64)> {
    let area_min_max: (f64, f64) = records.area.iter().cloned().fold((0.0, 0.0), |acc, x| {
        if x < acc.0 {
            (x, acc.1)
        } else if x > acc.1 {
            (acc.0, x)
        } else {
            acc
        }
    });
    let perimeter_min_max: (f64, f64) = records.perimeter.iter().cloned().fold((0.0, 0.0), |acc, x| {
        if x < acc.0 {
            (x, acc.1)
        } else if x > acc.1 {
            (acc.0, x)
        } else {
            acc
        }
    });
    let compactness_min_max: (f64, f64) = records.compactness.iter().cloned().fold((0.0, 0.0), |acc, x| {
        if x < acc.0 {
            (x, acc.1)
        } else if x > acc.1 {
            (acc.0, x)
        } else {
            acc
        }
    });
    let kernel_length_min_max: (f64, f64) = records.kernel_length.iter().cloned().fold((0.0, 0.0), |acc, x| {
        if x < acc.0 {
            (x, acc.1)
        } else if x > acc.1 {
            (acc.0, x)
        } else {
            acc
        }
    });
    let kernel_width_min_max: (f64, f64) = records.kernel_width.iter().cloned().fold((0.0, 0.0), |acc, x| {
        if x < acc.0 {
            (x, acc.1)
        } else if x > acc.1 {
            (acc.0, x)
        } else {
            acc
        }
    });
    let asymmetry_coefficient_min_max: (f64, f64) = records.asymmetry_coefficient.iter().cloned().fold((0.0, 0.0), |acc, x| {
        if x < acc.0 {
            (x, acc.1)
        } else if x > acc.1 {
            (acc.0, x)
        } else {
            acc
        }
    });
    let kernel_groove_length_min_max: (f64, f64) = records.kernel_groove_length.iter().cloned().fold((0.0, 0.0), |acc, x| {
        if x < acc.0 {
            (x, acc.1)
        } else if x > acc.1 {
            (acc.0, x)
        } else {
            acc
        }
    });

    vec![area_min_max, perimeter_min_max, compactness_min_max, kernel_length_min_max, kernel_width_min_max, asymmetry_coefficient_min_max, kernel_groove_length_min_max]
}

/// Normalizes the dataset using the min and max values for each column.
pub fn normalize(records: &Records) -> Vec<Vec<f64>> {
    let min_max = get_min_max(records);

    let mut area = vec![0.0; records.area.len()];
    let mut perimeter = vec![0.0; records.perimeter.len()];
    let mut compactness = vec![0.0; records.compactness.len()];
    let mut kernel_length = vec![0.0; records.kernel_length.len()];
    let mut kernel_width = vec![0.0; records.kernel_width.len()];
    let mut asymmetry_coefficient = vec![0.0; records.asymmetry_coefficient.len()];
    let mut kernel_groove_length = vec![0.0; records.kernel_groove_length.len()];
    let type_of_wheat = records.type_of_wheat.iter().map(|x| *x as f64).collect::<Vec<f64>>();

    for (i, record) in records.area.iter().enumerate() {
        area[i] = (record - min_max[0].0) / (min_max[0].1 - min_max[0].0);
    }
    for (i, record) in records.perimeter.iter().enumerate() {
        perimeter[i] = (record - min_max[1].0) / (min_max[1].1 - min_max[1].0);
    }
    for (i, record) in records.compactness.iter().enumerate() {
        compactness[i] = (record - min_max[2].0) / (min_max[2].1 - min_max[2].0);
    }
    for (i, record) in records.kernel_length.iter().enumerate() {
        kernel_length[i] = (record - min_max[3].0) / (min_max[3].1 - min_max[3].0);
    }
    for (i, record) in records.kernel_width.iter().enumerate() {
        kernel_width[i] = (record - min_max[4].0) / (min_max[4].1 - min_max[4].0);
    }
    for (i, record) in records.asymmetry_coefficient.iter().enumerate() {
        asymmetry_coefficient[i] = (record - min_max[5].0) / (min_max[5].1 - min_max[5].0);
    }
    for (i, record) in records.kernel_groove_length.iter().enumerate() {
        kernel_groove_length[i] = (record - min_max[6].0) / (min_max[6].1 - min_max[6].0);
    }

    let mut normalized_dataset = Vec::new();

    for (index, area_value) in area.iter_mut().enumerate() {
        normalized_dataset.push(vec![*area_value, perimeter[index], compactness[index], kernel_length[index], kernel_width[index], asymmetry_coefficient[index], kernel_groove_length[index], type_of_wheat[index]]);
    }

    normalized_dataset
}

pub fn main() {
    let dataset = read_wheat_csv("model_data/wheat.csv");
    let normalized_dataset = normalize(&dataset);

    let outputs = dataset.type_of_wheat;

    let mut nn_config = NeuralNetConfig::default();
    nn_config.set_hidden_nodes();

    let scores = run_network(&normalized_dataset, outputs, nn_config);
    let mean_accuracy = scores.iter().sum::<f64>() / scores.len() as f64;

    println!("Scores: ");
    for score in scores.iter() {
        print!("{}%, ", score);
    }
    println!();

    println!("Mean Accuracy: {}%", mean_accuracy);
}