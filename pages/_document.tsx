import NextDocument, { Html, Head, Main, NextScript } from 'next/document'
import React from 'react'

type Props = {}

class Document extends NextDocument<Props> {
  render() {
    return (
      <Html>
        <Head>
          <link href="http://fonts.cdnfonts.com/css/helvetica-255" rel="stylesheet" />
          <title>AI-ML Lab Programs</title>
          <meta name="title" content="AI-ML Lab Programs" />
          <meta name="description" content="VTU 7th Sem 2018 scheme, AI-ML Lab Programs." />
          <meta property="og:type" content="website" />
          <meta property="og:url" content="https://aiml.vercel.app" />
          <meta property="og:title" content="AI-ML Lab Programs" />
          <meta property="og:description" content="VTU 7th Sem 2018 scheme, AI-ML Lab Programs." />
          <meta property="og:image" content="https://i.ibb.co/Kqw19Hq/bird.png" />

          <meta property="twitter:card" content="summary_large_image" />
          <meta property="twitter:url" content="https://aiml.vercel.app" />
          <meta property="twitter:title" content="AI-ML Lab Programs" />
          <meta property="twitter:description" content="VTU 7th Sem 2018 scheme, AI-ML Lab Programs." />
          <meta property="twitter:image" content="https://i.ibb.co/Kqw19Hq/bird.png" />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}

export default Document