import { NextPage } from "next";
import { useRouter } from 'next/router';
import { progs } from '../data/progs';
import styles from '../styles/Home.module.css';
import Head from 'next/head';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import BackButton from "../components/back.component";
import Typed from "react-typed"

const Program: NextPage = () => {
    const router = useRouter();
    const { id } = router.query;
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
                    <Typed
                        strings={[`Prog ${id}: Not Found`, `Prog ${id}: Missing`, `Prog ${id}: Error 404`]}
                        typeSpeed={40}
                        backSpeed={50}
                        backDelay={2000}
                        loop
                        style={{
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

export default Program;