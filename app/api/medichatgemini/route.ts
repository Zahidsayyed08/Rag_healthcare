import { queryPineconeVectorStore } from "@/utils";
import { Pinecone } from "@pinecone-database/pinecone";
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { generateText, Message, StreamData, streamText } from "ai";
import { topK } from "@/app/config";

// Allow streaming responses up to 30 seconds
export const maxDuration = 60;
// export const runtime = 'edge';

const pinecone = new Pinecone({
    apiKey: "pcsk_757Uux_JWP3Vvx82ML38JDSPxhT7sHUyyv22P8sPqw7VbFktS8C8B2fEVswRYFue4HeCuw",
});

const google = createGoogleGenerativeAI({
    baseURL: 'https://generativelanguage.googleapis.com/v1beta',
    apiKey: "AIzaSyAtDFMTXL-SH_aE3SIGCo6oUEq-9cNm2To"
});

// gemini-1.5-pro-latest
// gemini-1.5-pro-exp-0801
const model = google('models/gemini-1.5-flash', {
    safetySettings: [
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' }
    ],
});

export async function POST(req: Request, res: Response) {
    const reqBody = await req.json();
    console.log(reqBody);

    const messages: Message[] = reqBody.messages;
    const userQuestion = `${messages[messages.length - 1].content}`;

    const reportData: string = reqBody.data.reportData;
    const query = `Represent this for searching relevant passages: patient medical report says: \n${reportData}. \n\n${userQuestion}`;

    // Pass topK parameter to the query
    const retrievals = await queryPineconeVectorStore(pinecone, 'index-one', "testspace", query, topK);

    const finalPrompt = `Here is a summary of a patient's clinical report, and a user query. Some generic clinical findings are also provided that may or may not be relevant for the report.
  Go through the clinical report and answer the user query.
  Ensure the response is factually accurate, and demonstrates a thorough understanding of the query topic and the clinical report.
  Before answering you may enrich your knowledge by going through the provided clinical findings. 
  The clinical findings are generic insights and not part of the patient's medical report. Do not include any clinical finding if it is not relevant for the patient's case.

  \n\n**Patient's Clinical report summary:** \n${reportData}. 
  \n**end of patient's clinical report** 

  \n\n**User Query:**\n${userQuestion}?
  \n**end of user query** 

  \n\n**Generic Clinical findings:**
  \n\n${retrievals}. 
  \n\n**end of generic clinical findings** 

  \n\nProvide thorough justification for your answer.
  \n\n**Answer:**
  `;

    const data = new StreamData();
    data.append({
        retrievals: retrievals
    });

    const result = await streamText({
        model: model,
        prompt: finalPrompt,
        onFinish() {
            data.close();
        }
    });

    return result.toDataStreamResponse({ data });
}

