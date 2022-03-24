import type { NextPage } from 'next'
import Head from 'next/head'
import Link from 'next/link'
import { progs } from '../data/progs'
import styles from '../styles/Home.module.css'
import Typed from "react-typed"
import { useEffect } from 'react'

const Home: NextPage = () => {
  useEffect(() => {
    console.log('happy hacking!')
  }, []);
  return (
    <div className={styles.container}>
      <Head>
        <title>AIML | Programs</title>
      </Head>

      <main className={styles.main}>
        <h1>AI ML Programs</h1>
        <Typed
          strings={["VTU 2018 Scheme", "7th Semester", "CSE"]}
          typeSpeed={40}
          backSpeed={50}
          backDelay={3000}
        // loop
        />
        <hr />
        <div className={styles.grid}>
          {
            Object.keys(progs).sort((a, b) => progs[a].id > progs[b].id ? 1 : -1).map((key) => {
              return (
                <GridItem
                  key={key}
                  title={"Prog " + progs[key].id}
                  data={progs[key].name}
                  id={key}
                />
              )
            })
          }
        </div>
        <div
        onClick={() => {
          window.open("", "_blank");
        }}
          style={{
            cursor: 'pointer',
            backgroundColor: '#f4f4f4',
            padding: '4px 10px',
            color: "black",
            fontSize: "0.8em",
            borderRadius: "2px",
            width: "fit-content",
            marginTop: "1.2rem",
          }}>
          Py Folder Alt
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
        <p className={styles.underLine}>{title}</p>
        <p className={styles.underLine}>{data}</p>
        <p style={{ color: "var(--color)", fontSize: "0.8rem" }}>
          See Code &rarr;
        </p>
      </div>
    </Link>
  )
}
