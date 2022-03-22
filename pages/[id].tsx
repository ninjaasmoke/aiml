import { NextPage } from "next";
import { useRouter } from 'next/router';
import { progs } from '../data/progs';
import styles from '../styles/Home.module.css';
import Head from 'next/head';
import Image from 'next/image';

import HeadImg from "../public/head.png";

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import vscDarkPlus from 'react-syntax-highlighter/dist/cjs/styles/prism/vsc-dark-plus';
import ghcolors from 'react-syntax-highlighter/dist/cjs/styles/prism/ghcolors';
import BackButton from "../components/back.component";
import Typed from "react-typed"

const Program: NextPage = () => {
    const router = useRouter();
    const { id } = router.query;
    if (id && typeof id === 'string' && id.includes("j")) return (
        <div className={styles.jupyter}>
            <Head>
                <title>Jupyter Notebook</title>
            </Head>

            <main>
                <section className={styles.headImg}>
                    <Image src={HeadImg} />
                </section>
                <div className={styles.jupCode}>
                    <div className={styles.jupInner}>
                        <div className={styles.jupInnerBar}>
                            In [1]:
                        </div>
                        <div className={styles.jupInnerCode}>
                            <SyntaxHighlighter language="python" style={ghcolors} showLineNumbers customStyle={{
                                userSelect: 'text',
                                border: '1px solid #d1d1d1',
                                borderRadius: '2px',
                            }}>
                                {
                                    codeA(id.charAt(0))
                                }
                            </SyntaxHighlighter>
                            <pre>
                                {outputA(id.charAt(0))}
                            </pre>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
    return (
        <div className={styles.prog}>
            {
                codeA(id).length > 0 &&
                <>
                    <Head>
                        <title>{nameA(id)}</title>
                        <meta name="description" content={"Code for " + nameA(id) + " in Python."} />
                    </Head>

                    <BackButton />

                    <h1>Program {id}</h1>
                    <p style={{
                        color: 'var(--accent)',
                    }}>
                        {nameA(id)}
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
                    <button style={{
                        cursor: 'pointer',
                        borderRadius: '4px',
                        outline: 'none',
                        border: 'none',
                        marginBottom: '1rem',
                    }}
                    onClick={() => {
                        router.push(`${id}j`);
                    }}
                    >Go to Jupyter Style</button>
                    <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers customStyle={{
                        borderRadius: '8px',
                        padding: '40px 8px',
                        userSelect: 'text',
                    }}>
                        {
                            codeA(id)
                        }
                    </SyntaxHighlighter>
                </>
            }
            {
                codeA(id).length === 0 &&
                <>
                    <Head>
                        <title>404 Error | Not found</title>
                    </Head>
                    <Typed
                        strings={[`Prog ${id}: Not Found`, `Prog ${id}: Missing`, `Prog ${id}: Error 404`]}
                        typeSpeed={40}
                        backSpeed={50}
                        backDelay={2000}
                        loop
                        style={{
                            fontWeight: 'bold',
                            color: 'var(--accent)',
                            position: 'absolute',
                            bottom: '0',
                            transform: 'translateY(-200%)',
                        }}
                    />
                </>
            }
        </div>
    );
}


const codeA = (id: string | string[] | undefined) => {
    try {
        if (id && (typeof id === "string") && progs[id] && progs[id]["code"])
            return progs[id]["code"];
        return "";
    } catch (error) {
        return "";
    }
}

const nameA = (id: string | string[] | undefined) => {
    try {
        if (id && (typeof id === "string") && progs[id] && progs[id]["name"])
            return progs[id]["name"];
        return "";
    } catch (error) {
        return "";
    }
}

const outputA = (id: string | string[] | undefined) => {
    try {
        if (id && (typeof id === "string") && progs[id] && progs[id]["output"])
            return progs[id]["output"];
        return "";
    } catch (error) {
        return "";
    }
} 

export default Program;