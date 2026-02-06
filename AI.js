import fetch from "node-fetch";
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";
import fs from "fs";
import path from "path";
import { Pinecone } from "@pinecone-database/pinecone";
import _ from "lodash";

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const INDEX_NAME = "my-rag-index";

let embedder;
let embedderReady = false;

async function initEmbedder() {
  console.log("[DEBUG] Initializing Xenova embedder...");
  const { pipeline } = await import("@xenova/transformers");
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  embedderReady = true;
  console.log("[DEBUG] Embedder ready");
}

async function ensureEmbedder() {
  if (!embedderReady) await initEmbedder();
}

async function embed(text) {
  console.log("[DEBUG] Embedding text:", text);
  await ensureEmbedder();
  const embeddings = await embedder(text);
  const arr = Array.from(embeddings.data);
  let vector;
  if (embeddings.dims.length === 3) {
    const tokens = embeddings.dims[1], hidden = embeddings.dims[2];
    vector = new Array(hidden).fill(0);
    for (let t = 0; t < tokens; t++) {
      for (let h = 0; h < hidden; h++) {
        vector[h] += arr[t * hidden + h];
      }
    }
    vector = vector.map(v => v / tokens);
  } else {
    vector = arr;
  }

  if (vector.length !== 256) {
    const down = [];
    for (let i = 0; i < 256; i++) {
      down[i] = vector[Math.floor(i * (vector.length / 256))];
    }
    vector = down;
  }



  console.log("[DEBUG] Embedding vector length:", vector.length);
  return vector;
}

async function queryContext(prompt, topK = 10, numSentences = 3) {
  console.log("[DEBUG] Querying RAG context...");
  await ensureEmbedder();

  try {
    const index = pinecone.index(INDEX_NAME);
    const embedding = await embed(prompt);

    const result = await index.query({
      vector: embedding,
      topK,
      includeMetadata: true,
    });

    if (!result.matches?.length) {
      console.log("[DEBUG] No matches returned from Pinecone");
      return "";
    }

    console.log("[DEBUG] Top matches returned from Pinecone:");
    result.matches.forEach((m, i) => {
      console.log(`  [${i + 1}] id: ${m.id}, score: ${m.score.toFixed(4)}, text: ${m.metadata?.story_text?.slice(0, 100)}${m.metadata?.story_text?.length > 100 ? "..." : ""}`);
    });

    const strongMatches = result.matches
      .filter(m => typeof m.score === "number" && m.score >= 0.2)
      .sort((a, b) => b.score - a.score);

    if (!strongMatches.length) {
      console.log("[DEBUG] No strong matches, skipping RAG");
      return "";
    }

    const chosen = strongMatches[0];
    const text = chosen.metadata?.story_text;

    if (!text) {
      console.log("[DEBUG] Chosen match has no usable text");
      return "";
    }

    const sentences = text
      .split(/(?<=[.?!])\s+/)
      .map(s => s.trim())
      .filter(Boolean);

    const context = sentences.slice(0, numSentences).join(" ");

    console.log("[DEBUG] Selected context:\n", context);
    return context;

  } catch (err) {
    console.error("[ERROR] Failed to query context:", err.message);
    return "";
  }
}


async function callLLM(prompt, context) {
  console.log("[DEBUG] Calling LLM...");
  console.log("[DEBUG] LLM Input - Prompt:", prompt);
  console.log("[DEBUG] LLM Input - Context:", context);
  console.log("[DEBUG] LLM Input - Context length:", context?.length || 0);

  const storyPhrases = [
  "Once upon a time", "Long ago", "Long, long ago", "In a faraway land", "In a distant kingdom",
  "There was once", "A long time ago", "In an age forgotten", "Before the world was as it is now",
  "In a time of magic", "In the beginning", "It was a dark and stormy night", "On a night like any other",
  "On a day unlike any other", "Not so long ago"
];


  const titlePhrases = [
    "The Tale of", "The Story of", "A Chronicle of", "The Legend of",
    "The Journey of", "A Saga of", "The Mystery of", "The Forgotten Story of",
    "The Adventures of", "The Chronicles of"
  ];

  const storyPhrase = storyPhrases[Math.floor(Math.random() * storyPhrases.length)];
  const titlePrefix = titlePhrases[Math.floor(Math.random() * titlePhrases.length)];

  const response = await fetch(`http://${process.env.AZURE_IP_ADDRESS}:8000/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: prompt, context: context, storyPhrase: storyPhrase, title_prefix: titlePrefix }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`LLM request failed: ${response.status} ${text}`);
  }

  const data = await response.json();
  if (!data.story || !data.title) {
    throw new Error("LLM returned empty story or title");
  }

  console.log("[DEBUG] LLM output received: ", { title: data.title, storyLength: data.story });
  return { story: data.story, title: data.title };
}

async function makeLeonardoMotionVideo(storyText) {
  console.log("[DEBUG] Requesting Leonardo motion video...");
  const res = await fetch(
    "https://cloud.leonardo.ai/api/rest/v1/generations-text-to-video",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.LEONARDO_API_KEY}`,
      },
      body: JSON.stringify({
        prompt: storyText,
        width: 832,
        height: 480,
        resolution: "RESOLUTION_480",
        model: "MOTION2FAST",
        frameInterpolation: true,
        isPublic: true,
        promptEnhance: true,
      }),
    }
  );

  if (!res.ok) throw new Error(`Leonardo request failed: ${res.status}`);
  const data = await res.json();
  const motionGenId = data.motionVideoGenerationJob?.generationId || data.generationId || data.id;
  if (!motionGenId) throw new Error("No generationId returned");

  const pollUrl = `https://cloud.leonardo.ai/api/rest/v1/generations/${motionGenId}`;
  let videoUrl = null;

  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 3000));
    const pollRes = await fetch(pollUrl, { headers: { Authorization: `Bearer ${process.env.LEONARDO_API_KEY}` } });
    if (!pollRes.ok) continue;
    const pollData = await pollRes.json();
    const gen = pollData.generations_by_pk;
    if (!gen) continue;
    console.log(`[DEBUG] Poll attempt ${i + 1}: status = ${gen.status}`);
    if (gen.status === "FAILED") throw new Error("Leonardo Motion job failed");
    if (gen.status !== "COMPLETE") continue;
    if (gen.generated_images?.length > 0) {
      const img = gen.generated_images[0];
      videoUrl = img.motionMP4URL || img.motionMp4Url || img.video || img.video_url;
      if (videoUrl) break;
    }
  }

  if (!videoUrl) throw new Error("Failed to get Motion video URL");
  console.log("[DEBUG] Leonardo Motion video URL:", videoUrl);
  return videoUrl;
}

ffmpeg.setFfmpegPath(ffmpegPath);

async function videoUrlToBase64(videoUrl, timeInSeconds = 2) {
  console.log("[DEBUG] Generating video thumbnail...");
  const outputPath = path.join(process.cwd(), `thumbnail_${Date.now()}.jpg`);

  const videoPath = path.join(process.cwd(), `video_${Date.now()}.mp4`);
  const videoRes = await fetch(videoUrl);
  const buffer = Buffer.from(await videoRes.arrayBuffer());
  fs.writeFileSync(videoPath, buffer);
  console.log("[DEBUG] Video downloaded to", videoPath);

  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .setFfmpegPath(ffmpegPath)
      .screenshots({
        timestamps: [timeInSeconds],
        filename: path.basename(outputPath),
        folder: path.dirname(outputPath),
        size: "320x240",
      })
      .on("end", () => {
        try {
          const imgBuffer = fs.readFileSync(outputPath);
          const base64 = `data:image/jpeg;base64,${imgBuffer.toString("base64")}`;
          fs.unlinkSync(outputPath);
          fs.unlinkSync(videoPath);
          console.log("[DEBUG] Thumbnail generated successfully");
          resolve(base64);
        } catch (err) {
          reject(err);
        }
      })
      .on("error", reject);
  });
}

export async function generate(prompt) {
  console.log("[DEBUG] Starting full generation pipeline...");
  const context = await queryContext(prompt);
  console.log("[DEBUG] Retrieved context:\n", context);
  const { title, story } = await callLLM(prompt, context);
  const videoUrl = await makeLeonardoMotionVideo(title);
  const previewImage = await videoUrlToBase64(videoUrl);
  console.log("[DEBUG] Full generation complete");
  return { title, story, videoUrl, previewImage };
}
