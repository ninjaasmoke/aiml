import type { NextPage } from 'next'
import Head from 'next/head'
import Link from 'next/link'
import styles from '../styles/Home.module.css'

const Home: NextPage = () => {
  return (
    <div className={styles.container}>
      <Head>
        <title>AIML | Programs</title>
      </Head>

      <main className={styles.main}>
        <h1>AI ML Programs</h1>
        <hr />
        <div className={styles.grid}>
          {
            <GridItem 
              title="Prog 6"
              data="Naive Bayes Classifier"
              id="6"
            />
          }
          {
            <GridItem
              title="Prog 7"
              data="K Means Clustering"
              id="7"
            />
          }
          {
            <GridItem
              title="Prog 8"
              data="K Nearest Neighbors"
              id="8"
            />
          }
        </div>
      </main>
    </div>
  )
}

export default Home

const GridItem = ({ title, data, id }: { title: string, data: string, id: string }) => {
  return (
    <Link href={id}>
      <div className={styles.gridItem}>
        <p>{title}</p>
        <p>
          {data}
        </p>
      </div>
    </Link>
  )
}
