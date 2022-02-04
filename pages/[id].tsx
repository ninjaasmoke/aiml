import { NextPage } from "next";
import { useRouter } from 'next/router';
import { progs } from '../data/progs';
import styles from '../styles/Home.module.css';
import Head from 'next/head';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import BackButton from "../components/back.component";

const Program: NextPage = () => {
    const router = useRouter();
    const { id } = router.query;
    return (
        <div className={styles.prog}>

            <Head>
                <title>{nameA(id)}</title>
                <meta name="description" content={"Code for " + nameA(id) + " in Python."} />
            </Head>

            <BackButton />

            <h1>Program {id}</h1>
            <p style={{
                color: 'var(--accent)',
            }}>
                {id && (typeof id === "string") && progs[id]["name"]}
                <button className={styles.copy}
                    onClick={async () => {
                        await navigator.clipboard.writeText(codeA(id));
                    }}
                >
                    Copy
                </button>
            </p>
            <p>
                or <code className={styles.instr}>CTRL + A</code> and <code className={styles.instr}>CTRL + C</code> to copy code
            </p>
            <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{
                borderRadius: '8px',
                padding: '40px 8px',
                userSelect: 'text',
            }}>
                {
                    codeA(id)
                }
            </SyntaxHighlighter>
        </div>
    );
}


const codeA = (id: string | string[] | undefined) => {
    if (id && (typeof id === "string") && progs[id]["code"])
        return progs[id]["code"];
    return "";
}

const nameA = (id: string | string[] | undefined) => {
    if (id && (typeof id === "string") && progs[id]["name"])
        return progs[id]["name"];
    return "";
}

export default Program;