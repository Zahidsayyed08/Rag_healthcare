//app/api/chat/route.ts
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
import { NextResponse } from 'next/server';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);

function createSystemPrompt(medicineData: any, source: 'FDA' | 'LLM') {
  return `
You are a knowledgeable medical information assistant. You provide factual information about medicines based on official data. You should:

MEDICINE CONTEXT:
Using verified ${source === 'FDA' ? 'FDA' : 'medical'} data for ${medicineData.name}:
- Name: ${medicineData.name}
- Generic Name: ${medicineData.genericName}
- Description: ${medicineData.description}
- Indications: ${medicineData.indications}
- Warnings: ${medicineData.warnings}
- Dosage: ${medicineData.dosage}

YOUR APPROACH:
1. Share information from the official data clearly and directly
2. Explain medical terms in simple language
3. Include relevant safety information when appropriate
4. If asked about something not in the data, explain what information is available
5. For specific medical situations, recommend consulting a healthcare provider while still sharing the general information available
6. Maintain a helpful, informative tone while being factual
7. Share dosage information from the official data, adding that a healthcare provider can give specific guidance
8. Use bullet points for clarity when listing multiple points

Remember: While recommending consulting healthcare providers when needed, you should still provide the factual information available in the medicine data.
`;
}

export async function POST(request: Request) {
  try {
    const { messages, medicineData } = await request.json();
    
    if (!medicineData) {
      return NextResponse.json({ error: 'Medicine data is required' }, { status: 400 });
    }

    // Determine data source
    const dataSource = medicineData.id?.startsWith('gemini-') ? 'LLM' : 'FDA';
    const systemPrompt = createSystemPrompt(medicineData, dataSource);

    // Format messages for Gemini
    const formattedMessages = [
      {
        role: 'model',
        parts: [{ text: systemPrompt }]
      },
      ...messages.map((message: any) => ({
        role: message.role === 'user' ? 'user' : 'model',
        parts: [{ text: message.content }]
      }))
    ];

    // Initialize Gemini model with more conversational settings
    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-flash",
      generationConfig: {
        temperature: 0.7, // Increased for more natural responses
        maxOutputTokens: 1000,
        topK: 40,
        topP: 0.8,
      }
    });

    // Generate response
    const result = await model.generateContent({
      contents: formattedMessages,
      safetySettings: [{
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      }]
    });

    const response = await result.response;
    const text = response.text();

    return NextResponse.json({ response: text });
    
  } catch (error) {
    console.error('Chat error:', error);
    return NextResponse.json(
      { 
        error: 'Failed to process chat message',
        details: error instanceof Error ? error.message : 'Unknown error'
      }, 
      { status: 500 }
    );
  }
}