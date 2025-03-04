'use client'

import Link from 'next/link'
import './home-upload.css'

export default function Home() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-blue-100 to-white home-page">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-8 text-blue-800">Document assistant service</h1>
        <div className="space-y-4">
          <Link href="/chatbot" className="inline-block px-6 py-3 bg-blue-500 text-white font-semibold rounded-lg shadow-md hover:bg-blue-600 transition duration-300 ease-in-out transform hover:-translate-y-1">
            Go to Chatbot
          </Link>
          <br />
          <Link href="/uploadservice" className="inline-block px-6 py-3 bg-green-500 text-white font-semibold rounded-lg shadow-md hover:bg-green-600 transition duration-300 ease-in-out transform hover:-translate-y-1">
            Go to Upload Service
          </Link>
        </div>
      </div>
    </main>
  )
}