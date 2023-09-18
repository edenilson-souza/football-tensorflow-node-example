import * as tf from "@tensorflow/tfjs-node";
import { tensor2d } from "@tensorflow/tfjs-node";

// Dados históricos de partidas de futebol
const matches = [
    { homeScore: 32, awayScore: 30, result: "Win" }, //Adicione mais estatísticas aqui...
    { homeScore: 30, awayScore: 32, result: "Loss" },
    { homeScore: 30, awayScore: 30, result: "Loss" },
    { homeScore: 30, awayScore: 40, result: "Loss" },
    { homeScore: 20, awayScore: 15, result: "Win" },
    { homeScore: 15, awayScore: 20, result: "Loss" },
    { homeScore: 20, awayScore: 20, result: "Loss" },
    { homeScore: 26, awayScore: 25, result: "Win" }
    // Adicione mais partidas aqui...
];

// Preparando os dados
const inputData: number[][] = [];
const outputData: number[] = [];

for (const match of matches) {
    const homeScore = match.homeScore;
    const awayScore = match.awayScore;
    //Preparando os dados de entrada e saída
    //Se adicionar mais estatísticas, adicione mais valores aqui
    const result = match.result === "Win" ? 1 : 0; // Codificar 'Win' como 1 e 'Loss' como 0

    inputData.push([homeScore, awayScore]); // Adicionando os dados de entrada
    outputData.push(result); // Adicionando os dados de saída
}

const xData = tensor2d(inputData);
const yData = tensor2d(outputData, [outputData.length, 1]);

// Criando o modelo de Regressão Logística
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2], activation: "sigmoid" }));

// Compilando o modelo
model.compile({ optimizer: tf.train.adam(), loss: "binaryCrossentropy", metrics: ["accuracy"] });

// Treinando o modelo
async function trainModel() {
    await model
        .fit(xData, yData, {
            epochs: 10000,
            verbose: 1,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch: any, logs: any) => {
                    console.log(`Epoch ${epoch + 1}/${100} - loss: ${logs?.loss.toFixed(4)} - acc: ${logs?.acc.toFixed(4)}`);
                }
            }
        })
        .then(() => {
            console.log("Treinamento concluído.");
            model.summary();
            //SAVE
            model.save("file://./model").then(() => {
                console.log("Modelo salvo.");
            });
        });
}

trainModel();
