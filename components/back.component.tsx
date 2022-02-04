import BackIcon from '../public/back.png';
import { useRouter } from 'next/router';

export default function BackButton() {
    const router = useRouter();

    return (
        <span onClick={() => {
            router.push('/');
        }}
            style={{
                cursor: 'pointer',
            }}
        >
            <img src={BackIcon.src} alt="Back Button" height={20} width={20} />
        </span>
    )
}