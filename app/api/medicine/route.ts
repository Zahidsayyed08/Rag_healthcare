import { NextResponse } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'

const FDA_API_KEY = process.env.FDA_API_KEY
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY

const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY || '')

async function searchFDA(query: string) {
  try {
    const response = await fetch(
      `https://api.fda.gov/drug/label.json?api_key=${FDA_API_KEY}&search=openfda.brand_name:"${encodeURIComponent(query)}" OR openfda.generic_name:"${encodeURIComponent(query)}"&limit=5`
    )
    
    if (!response.ok) {
      throw new Error(`FDA API responded with status ${response.status}`);
    }

    const data = await response.json()
    return data.results.map((result: any) => ({
      id: result.id,
      name: result.openfda.brand_name?.[0] || 'Unknown',
      genericName: result.openfda.generic_name?.[0] || 'Unknown',
      description: result.description?.[0] || 'No description available',
      indications: result.indications_and_usage?.[0] || 'No indications available',
      warnings: result.warnings?.[0] || 'No warnings available',
      dosage: result.dosage_and_administration?.[0] || 'No dosage information available',
    }))
  } catch (error) {
    console.error('Error fetching from FDA:', error)
    return null
  }
}

async function searchGemini(query: string) {
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" })
    
    const prompt = `Provide detailed medical information about "${query}" in this exact format:

Name: [Medicine Name]
Generic Name: [Generic Name if available, otherwise "Unknown"]
Description: [Detailed description of what this medicine is]
Indications: [What conditions this medicine treats]
Warnings: [Important safety warnings and side effects]
Dosage: [General dosage guidelines]

Please ensure all information is evidence-based and follows medical guidelines. If this exact medicine isn't known, provide information about the closest known similar medicine or drug class, but clearly state this fact in the description.`

    const result = await model.generateContent([
      { text: prompt }
    ])
    
    const response = await result.response
    const text = response.text()

    // Parse the generated text into an object
    const sections = text.split('\n')
    const medicineInfo: Record<string, string> = {}
    
    sections.forEach(section => {
      const [key, ...valueParts] = section.split(': ')
      if (key && valueParts.length > 0) {
        const cleanKey = key.toLowerCase().replace(/[^a-z]/g, '')
        medicineInfo[cleanKey] = valueParts.join(': ').trim()
      }
    })

    return [{
      id: `gemini-${Date.now()}`,
      name: medicineInfo.name || query,
      genericName: medicineInfo.genericname || 'Unknown',
      description: medicineInfo.description || 'No description available',
      indications: medicineInfo.indications || 'No indications available',
      warnings: medicineInfo.warnings || 'No warnings available',
      dosage: medicineInfo.dosage || 'No dosage information available',
    }]
  } catch (error) {
    console.error('Error fetching from Gemini:', error)
    return null
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const query = searchParams.get('query')

  if (!query) {
    return NextResponse.json(
      { error: 'Query parameter is required' }, 
      { status: 400 }
    )
  }

  let medicines = await searchFDA(query)

  if (!medicines || medicines.length === 0) {
    medicines = await searchGemini(query)
  }

  if (!medicines || medicines.length === 0) {
    return NextResponse.json(
      { error: 'No medicine information found' }, 
      { status: 404 }
    )
  }

  return NextResponse.json(medicines)
}