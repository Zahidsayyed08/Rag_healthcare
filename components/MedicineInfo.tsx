import ReactMarkdown from 'react-markdown'

interface Medicine {
  id?: string;
  name: string;
  genericName?: string;
  description: string;
  indications: string;
  warnings: string;
  dosage: string;
}

interface MedicineInfoProps {
  medicine: Medicine;
}

export default function MedicineInfo({ medicine }: MedicineInfoProps) {
  return (
    <div className="space-y-4 prose">
      <h2 className="text-2xl font-semibold">{medicine.name}</h2>
      {medicine.genericName && (
        <p className="text-gray-600">Generic Name: {medicine.genericName}</p>
      )}
      <div>
        <h3 className="text-lg font-semibold">Description</h3>
        <ReactMarkdown>{medicine.description}</ReactMarkdown>
      </div>
      <div>
        <h3 className="text-lg font-semibold">Indications and Usage</h3>
        <ReactMarkdown>{medicine.indications}</ReactMarkdown>
      </div>
      <div>
        <h3 className="text-lg font-semibold">Warnings</h3>
        <ReactMarkdown>{medicine.warnings}</ReactMarkdown>
      </div>
      <div>
        <h3 className="text-lg font-semibold">Dosage and Administration</h3>
        <ReactMarkdown>{medicine.dosage}</ReactMarkdown>
      </div>
    </div>
  )
}